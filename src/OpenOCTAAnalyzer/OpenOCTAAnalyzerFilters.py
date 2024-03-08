# OCTAnalyzerFilters.py

"""
Open OCTA Analyzer v 1.0
(C) 2023 - 2024 by Claus von der Burchard

This library allows to automatically quantify and compare OCTA images
Please visit https://github.com/clausvdb/OpenOCTAAnalyzer for further information
"""


from skimage.filters import threshold_otsu, frangi
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
from skimage.morphology import skeletonize

class OpenOCTAAnalyzerFilters:

    @staticmethod
    def standardOtsu(image):
        """
        Applies an Otsu threshold
        """
        thresh = threshold_otsu(image)
        return image > thresh

    @staticmethod
    def standardFrangi(image):
        """
        Applies a frangi filter optimized for small vessels and performs a threshold
        """

        small_vessels = frangi(image, beta=0.5, gamma=15, black_ridges=False, sigmas=range(1,4,1))
        thres = threshold_otsu(small_vessels)

        return small_vessels > thres

    @staticmethod
    def standardFrangiSmallAndBig(image):
        """
        Applies a frangi filter optimized for both small and bigger vessels and performs a threshold
        """

        small_vessels = frangi(image, beta=0.5, gamma=15, black_ridges=False, sigmas=range(1,4,1))
        big_vessels = frangi(image, beta=0.5, gamma=15, black_ridges=False, sigmas=range(6,10,1))
        thres = threshold_otsu(small_vessels)
        thres2 = threshold_otsu(big_vessels)

        return np.logical_or(small_vessels > thres, big_vessels > thres2)

    @staticmethod
    def standardFrangiVLD(image):
        return skeletonize(OpenOCTAAnalyzerFilters.standardFrangi(image))

    @staticmethod
    def standardRaw(image):
        """
        Empty container function, returns input unaltered
        """
        return image #if np.max(image) <= 1 else image / 255

    @staticmethod
    def standardGradient(image):
        return np.abs(np.gradient(image, axis=0) + np.gradient(image, axis=1)) / 2

    @staticmethod
    def isolateBigVessels(image):
        frangi_img = frangi(image, black_ridges=False, sigmas=range(7,8,1), alpha=0.9)#

        thres2 = threshold_otsu(frangi_img)
        frangi_thres = frangi_img > thres2 / 2

        return frangi_thres

    @staticmethod
    def jermanVesselness2D(I, sigmas, spacing, tau, brightondark=False):
        """
        Calculates the vesselness probability map (local tubularity) of a 2D input image after the Jerman Method
        DEBUG STATUS!
        
        Args:
        - I: 2D image
        - sigmas: vector of scales on which the vesselness is computed
        - spacing: input image spacing resolution - during hessian matrix computation, the gaussian filter kernel size in each dimension can be adjusted to account for different image spacing for different dimensions
        - tau: (between 0.5 and 1): parameter that controls response uniformity - lower tau -> more intense output response            
        - brightondark: (bool, optional): are vessels (tubular structures) bright on dark background or dark on bright (default for 2D is False)
        
        Returns:
        - vesselness: maximum vesselness response over scales sigmas
        """


        def imageEigenvalues(I, sigma, spacing, brightondark):
            # calculate the 2 eigenvalues for each voxel in a 2D image
            Hxx, Hyy, Hxy = Hessian2D(I, sigma, spacing)
            # Correct for scaling
            c = sigma ** 2
            Hxx *= c
            Hxy *= c
            Hyy *= c
            # reduce computation by computing vesselness only where needed
            B1 = -(Hxx + Hyy)
            B2 = Hxx * Hyy - Hxy ** 2
            T = np.ones_like(B1)
            if brightondark:
                T[B1 < 0] = 0
                T[(B2 == 0) & (B1 == 0)] = 0
            else:
                T[B1 > 0] = 0
                T[(B2 == 0) & (B1 == 0)] = 0
            indeces = np.nonzero(T)
            Hxx = Hxx[indeces]
            Hyy = Hyy[indeces]
            Hxy = Hxy[indeces]
            # Calculate eigen values
            lambda1, lambda2 = eigvalOfHessian2D(Hxx, Hxy, Hyy)
            Lambda1 = np.zeros_like(T)
            Lambda2 = np.zeros_like(T)
            Lambda1[indeces] = lambda1
            Lambda2[indeces] = lambda2
            # some noise removal
            Lambda1[~np.isfinite(Lambda1)] = 0
            Lambda2[~np.isfinite(Lambda2)] = 0
            Lambda1[np.abs(Lambda1) < 1e-4] = 0
            Lambda2[np.abs(Lambda2) < 1e-4] = 0
            return Lambda1, Lambda2

        def Hessian2D(I, Sigma=1, spacing=(1, 1)):
            if Sigma > 0:
                F = gaussian_filter(I, sigma=Sigma, order=0, mode='reflect')
            else:
                F = I
                
            Dy = np.gradient(F, axis=0)
            Dyy = np.gradient(Dy, axis=0)
            Dx = np.gradient(F, axis=1)
            Dxx = np.gradient(Dx, axis=1)
            Dxy = np.gradient(Dx, axis=0)
            
            return Dxx, Dyy, Dxy


        def eigvalOfHessian2D(Dxx, Dxy, Dyy):
            # This function calculates the eigen values from the hessian matrix, sorted by abs value
            
            # Compute the eigenvectors of J, v1 and v2
            tmp = np.sqrt((Dxx - Dyy)**2 + 4*Dxy**2)
            
            # Compute the eigenvalues
            mu1 = 0.5*(Dxx + Dyy + tmp)
            mu2 = 0.5*(Dxx + Dyy - tmp)
            
            # Sort eigen values by absolute value abs(Lambda1)<abs(Lambda2)
            check = np.abs(mu1) > np.abs(mu2)
            Lambda1 = mu1.copy()
            Lambda2 = mu2.copy()
            Lambda1[check] = mu2[check]
            Lambda2[check] = mu1[check]
            
            return Lambda1, Lambda2

        verbose = 1
        
        I = I.astype(np.float32)
        
        for j in range(len(sigmas)):
            if verbose:
                print(f"Current filter scale (sigma): {sigmas[j]}")
            
            _, Lambda2 = imageEigenvalues(I, sigmas[j], spacing, brightondark)
            
            if brightondark:
                Lambda2 = -Lambda2
            
            # proposed filter at current scale
            Lambda3 = Lambda2
            
            Lambda_rho = Lambda3
            Lambda_rho[(Lambda3 > 0) & (Lambda3 <= tau * np.max(Lambda3))] = tau * np.max(Lambda3)
            Lambda_rho[Lambda3 <= 0] = 0
            
            response = Lambda2 * Lambda2 * (Lambda_rho - Lambda2) * 27 / (Lambda2 + Lambda_rho) ** 3
            
            response[(Lambda2 >= Lambda_rho / 2) & (Lambda_rho > 0)] = 1    
            response[(Lambda2 <= 0) | (Lambda_rho <= 0)] = 0
            response[~np.isfinite(response)] = 0
            
            if j == 0:
                vesselness = response
            else:
                vesselness = np.maximum(vesselness, response)
            
            del response, Lambda2, Lambda3
        
        vesselness = vesselness / np.max(vesselness) # should not be really needed   
        vesselness[vesselness < 1e-2] = 0
        
        return vesselness


