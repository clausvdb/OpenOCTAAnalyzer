# How To Annotate A Patient

To annotate a patient, all relevant slabs (e.g. SVP, ICP, and DCP) should be exported to one directory, preferably as .png.
All files must be exported in the same fashion, i.e. same size etc.
Then, we need to define a Localizer file based on one of the images. Choose any slab where you can both clearly identify the edges of the OCTA image and the center. Usually, the SVP is a good option for this.

We will modify three pixels:
- The center of the fovea should be marked with ONE green pixel
- The top left and bottom right edges of the OCTA image should be marked with one blue pixel each. The markings should be set in a fashion that the pixels are just outside of the OCTA image
- See Patient1 OD Localizer.png for an example