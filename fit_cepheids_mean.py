import os
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from photutils.psf import PSFPhotometry
from photutils.psf import extract_stars, SourceGrouper
from photutils.psf import FittableImageModel
from photutils.psf import IntegratedGaussianPRF
from photutils.psf.matching import resize_psf
from photutils.background import MMMBackground, MADStdBackgroundRMS, LocalBackground
from photutils.detection import DAOStarFinder
from astropy.modeling.fitting import LevMarLSQFitter
from photutils.aperture import CircularAperture, ApertureMask, CircularAnnulus, aperture_photometry
import matplotlib.pyplot as plt


def psf_match_hst(x, y):
    xpsf = np.array([346,1032,1719,2406,3092,3779])
    ypsf = np.array([366,1098,1829,2195,2561,3293,4024])
    xval = np.sum(x > xpsf)
    yval = np.sum(y > ypsf)
    id = yval*7 + xval
    return id    


# Load the list of targets and their sky coordinates
tab = Table.read('../n1015_ceph_cat.dat', format='ascii')
#tab=tab[tab['ID']==56181]
print(tab)
ID=tab['ID']
RA=tab['RAdeg']
DEC=tab['DEdeg']
Per=tab['Per']

# Set up the base for output table
tab_out=Table([ID, RA, DEC],names=('id', 'RA', 'DEC'))

# Define the size of the cutout region around each target
cutout_size = 15 # Adjust this value as needed
cntr=(cutout_size-1)/2

#Define the coordinate shift in degrees relative to old catalogue
dra = 4.5*0.04/3600.
dde = 2.0*0.04/3600.

# Set stat aperture radius in px
rad = 2.



# Load the PSF from psf.fits file
psf_hdu_160 = fits.open('../PSF/PSFSTD_WFC3IR_F160W.fits')
psf_hdu_555 = fits.open('../PSF/PSFSTD_WFC3UV_F555W.fits')
psf_hdu_814 = fits.open('../PSF/STDPBF_WFC3UV_F814W.fits')
psf_cube_160 = psf_hdu_160[0].data # 3x3
psf_cube_555 = psf_hdu_555[0].data # 7x8
psf_cube_814 = psf_hdu_814[0].data # 7x8 x 9
focus = 5 # choose position of HST focus

# AB to Vega conversion for F160W, F555W, F814W
ab_to_vega=np.array([0.0,0.0,0.0])

# Encircled energy in 10px aperture for F160W, F555W, F814W
EE10=np.array([0.833,0.90,0.90])

# Locate folders with epochs, must be in format ????nn where nn is 2 digit number of epoch i.e. 01 or 12
folder = glob.glob('ic*')

# Set up the fitting and background routines
bkgrms = MADStdBackgroundRMS()
bkg_estimator = MMMBackground()
daogroup = SourceGrouper(5.)
fitter = LevMarLSQFitter(calc_uncertainties=True)

# Array for output fluxes and errors and obs-times
mag_out=np.zeros((ID.size,3))
err_out=np.zeros((ID.size,3))


for b, band in enumerate(['F160W','F555W','F814W']):
    # Load the HST image with WCS information 
    data, sciheader= fits.getdata(band+'_total_sci.fits', header=True)
    data+=1.
    wcs = WCS(sciheader)   
    wht = fits.getdata(band+'_total_wht.fits', header=False)
    
    #Zeropoints
    flux_inf_to_10px = EE10[b]
    zpt_ab_10px_noapcorr = -2.5*np.log10(sciheader['PHOTFLAM'])-5*np.log10(sciheader['PHOTPLAM'])-2.408+2.5*np.log10(flux_inf_to_10px)
    zpt_v_10px_noapcorr = zpt_ab_10px_noapcorr + ab_to_vega[b]
    
    # array for psf phot
    flux_psf = ID.data*0.
    err_psf = ID.data*0. 
    resid_ap = ID.data*0.

    # Extract sky coordinates from the table of targets and ID numbers and correct for wcs offset
    target_coords = SkyCoord(ra=tab['RAdeg']-dra, dec=tab['DEdeg']-dde, unit='deg')

    # Convert sky coordinates to pixel coordinates 
    target_x, target_y = wcs.all_world2pix(target_coords.ra, target_coords.dec, 0)


    # Perform PSF photometry on each target
    for i, xarr in enumerate(target_x):

        # Determine x,y position of the target in the large image
        target_pix = [target_x[i], target_y[i]]

        # Choose the appropriate PSF for the target position and resize it to match the data sampling
        pos = psf_match_hst(target_x[i], target_y[i]) 
        if band == 'F160W':
            psf_data = psf_cube_160[focus, :, :]
        elif band == 'F555W':
            psf_data = psf_cube_555[pos, :, :]
        else: 
            psf_data= psf_cube_814[focus, pos, :, :]
            
        #aperture correction
        aperture_psf = CircularAperture((50,50), 40)
        psf_ap = aperture_photometry(psf_data, aperture_psf)
        ap_corr = psf_ap['aperture_sum']/np.nansum(psf_data)
        
        #zeropoint  CHECK SIGN !!!
        zpt_apcorr = zpt_v_10px_noapcorr-2.5*np.log10(ap_corr)
            
        psf = FittableImageModel(psf_data, normalize=True, x_0=50, y_0=50, oversampling=1)
        test = psf.normalized_data
        test_res = resize_psf(test, 1, 4, order=5)

        # Create a cutout around the target
        try:
            cutout = Cutout2D(data, target_pix, cutout_size, wcs=None)
            small_cutout = Cutout2D(data, target_pix, (3,3), wcs=None)
        except:
            cutout = Cutout2D(np.zeros((100,100)),(50,50),cutout_size, wcs=None)
            small_cutout = Cutout2D(np.zeros((100,100)), (50,50), (3,3), wcs=None)
        # Save the cutouts
        cutout_directory = 'cutouts/'
        cutout_filename = cutout_directory + f'cutout_{band}{ID[i]}.fits'
        fits.writeto(cutout_filename, cutout.data, overwrite=True)
        
        
        if np.count_nonzero(~np.isfinite(cutout.data)) < 0.3*cutout_size**2:
            # Estimate the background RMS in the cutout region
            #std = bkgrms(Cutout2D(data, target_pix, 11, wcs=None).data)
            std = bkgrms(cutout.data)   

            #Estimate threshold for DAOfind detection
            thresh = 0.7*std * (5.-3./((np.nanmax(small_cutout.data)-bkg_estimator(cutout.data))/0.03))+0.3*bkg_estimator(cutout.data)
            if thresh < 0.02:
                thresh = 0.02

            eureka = False
            idex=0
            #print(thresh, std, np.nanmax(small_cutout.data), bkg_estimator(cutout.data), ID[i])

            # This loop optimizes the detection threshold if no sources are initially found within 3px from the target

            while eureka==False:    
                daofind = DAOStarFinder(fwhm=2., threshold=thresh, sigma_radius=3., sharplo=0.0, sharphi=2.0, roundlo=-2.0, roundhi=2.0, exclude_border=True)
            # This try block is needed because for some values of thresh the find_stars routine crashes and I have no good explanation why
                try:
                    sources = daofind.find_stars(cutout.data)
                    dist_src = np.sqrt((sources['xcentroid']-cntr)**2+(sources['ycentroid']-cntr)**2)
                    if np.any(np.where(dist_src<3,True,False)):
                        eureka = True
                    else:
                        thresh-=0.001
                        idex+=1
                        print(idex)
                        if idex>=10:
                            eureka = True
                except:
                    thresh-=0.002




            #print(sources['xcentroid'],sources['ycentroid'],sources['peak'],sources['flux'])


            local_tab=Table([[cntr],[cntr]],names=('x_0','y_0'))

            #set local bkg estimator
            local_bkg_est=LocalBackground(3, 7, bkg_estimator)

            print(thresh, std, np.nanmax(small_cutout.data), bkg_estimator(cutout.data), ID[i])


            # Define the PSF model
            psf = FittableImageModel(test_res, normalize=True, x_0=17, y_0=17,oversampling=1)

            # Set up the PSF photometry task
            photometry = PSFPhotometry(finder=daofind, grouper=None, localbkg_estimator=None, psf_model=psf, fitter=fitter, fitter_maxiters=3000, fit_shape=(15,15), aperture_radius=2.)

            # Perform PSF photometry on the cutout
            brd_mask=np.ones(cutout.data.shape, dtype=bool)
            brd_mask[2:-2,2:-2]=False
            ceph_tab = photometry(cutout.data-bkg_estimator(cutout.data), init_params=None)
            print(ceph_tab['x_fit'],ceph_tab['y_fit'],ceph_tab['flux_fit'])

            # Compute the image residuals
            residuals = photometry.make_residual_image(cutout.data,(15,15))

            # Define the residuals filename
            output_directory = 'residuals/'
            output_filename = output_directory + f'residual_{band}{ID[i]}.fits'


            # Save the residuals as a FITS file
            fits.writeto(output_filename, residuals, overwrite=True)

            # aperture statistics
            aperture=CircularAperture((cntr,cntr), rad)
            ap_annulus=CircularAnnulus((cntr,cntr), rad, rad+2)
            phot_ap = aperture_photometry(residuals, aperture)
            phot_an = aperture_photometry(residuals, ap_annulus)
            resid_ap[i] = phot_ap['aperture_sum'] - phot_an['aperture_sum']/ap_annulus.area*aperture.area

            # Append flux of the centermost PSF
            if (ceph_tab is None) or (ceph_tab['x_fit'] is None) :
                flux_psf[i]=1e-4
                err_psf[i]=1e6  
                print(id)
            else:
                #print(ceph_tab['x_fit'])
                ceph_tab.add_column(np.sqrt((ceph_tab['x_fit']-cntr)**2+(ceph_tab['y_fit']-cntr)**2),name='dist')
                tmp_ind=np.argmin(ceph_tab['dist'])
                flux_tmp=ceph_tab['flux_fit']
                flux_psf[i]=flux_tmp[tmp_ind]

                if fitter.fit_info['param_cov'] is None:
                    err_psf[i]=0.

                else:
                    flux_err_tmp=ceph_tab['flux_err']
                    err_psf[i]=flux_err_tmp[tmp_ind]
                    if not np.isfinite(err_psf[i]):
                        err_psf[i]=1e3
        else:
            flux_psf[i] = np.nan
            err_psf[i] = np.nan
            
    #Crowding Bias
    mu = zpt_apcorr - 2.5*np.log10(bkg_estimator(cutout.data))
    c_bias = -0.013*mu**2 + 0.61*mu - 7.4
    print(flux_psf)        
    mag_psf = zpt_apcorr - 2.5*np.log10(flux_psf) + c_bias
    err_mag = 2.5*np.log10(1+err_psf/flux_psf)
    tab_out.add_column(mag_psf,name='mag_'+band)
    tab_out.add_column(err_mag,name='err_'+band)

#    tab_out.write('detected_sources_F160W.dat',format='ascii.commented_header', overwrite=True)

    resid_ap[i]/=flux_psf[i]

    mag_out[:,b] = mag_psf
    err_out[:,b] = err_mag

    figure = plt.figure(figsize=(24,15))
    plt.ylim(-0.5,1)
    plt.errorbar(ID, resid_ap, yerr=resid_ap*0, fmt='o', ms=15, mew=4, c='black',elinewidth=4,capsize=4, capthick=3)
    plt.xlabel("ID",size=35)
    plt.ylabel("residual",size=35)
    plt.tick_params(labelsize=32)
    plt.tight_layout()

    plt.savefig(f'stat_phot_{band}.png')
    plt.close()
    
    
    figure = plt.figure(figsize=(24,15))


tab_out.write('mean_magnitudes.dat',format='ascii.commented_header', overwrite=True)  

RH = 0.386
RI = 1.3
WH = mag_out[:,0] - RH*(mag_out[:,1]-mag_out[:,2])
WI = mag_out[:,2] - RI*(mag_out[:,1]-mag_out[:,2])

ymax = 27
ymin = 23

print(mag_out)


plt.errorbar(Per, WI , yerr=resid_ap*0, fmt='o', ms=15, mew=4, c='black',elinewidth=4,capsize=4, capthick=3)
plt.xlabel("Period, days",size=35)
plt.ylim(plt.ylim(ymax,ymin)[::-1])
plt.ylabel("Magnitude",size=35)
plt.tick_params(labelsize=32)
plt.tight_layout()

plt.savefig(f'PLR.png')
plt.close()



    


