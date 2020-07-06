from BScThesis import PatternMaker as pm, HelperFunctions as hf, DicomScripts as ds, ImageFilters as imfil
import os

############# Global Variables #############

ringthickness = 12   # mm

############# Function Definitions #########


# Filename without any file-ending, ringdiameter in mm
def analizeSingleDicom(dicomfilename, excelfile=""):

    # Import Dicom file and me tadata
    dicomfile = ds.import_experimental_dicom(dicomfilename)

    # Rotate Dicom file
    dicomfile.rot90cc()

    # dicomfile.setfilteredarray(imfil.gauslogmultithreaded(dicomfile.getarray(), 7, 4))
    # dicomfile.setfilteredarray(imfil.gauslogfilter(dicomfile.getarray(), 7, 4, gaus1D=False))
    dicomfile.setfilteredarray(imfil.morphological_filter(dicomfile.getarray()))

    ds.imageview3d(hf.cvcontour3D(dicomfile.getfilteredarray()), "Boundingbox_shit")

    # Show both arrays next to each other
    ds.imageview3d([dicomfile.getarray(), dicomfile.getfilteredarray()], dicomfile.getname())

    # Get patient for dicom file if provided
    if not excelfile == "":
        patient = hf.makePatientData(dicomfilename, excelfile)

        # Add patient to dicom object
        dicomfile.setpatient(patient)

    # Create pattern that will be searched for
    pattern2 = pm.circlePatternImage(13, dicomfile.slice_thickness, 6)

    # View pattern
    # hf.imageView2D(pattern2, "Pattern")

    convolvedvolume = hf.findPatternInImageTemplate(dicomfile.getfilteredarray(), pattern2)

    convolvednormalized = hf.normalize(convolvedvolume)

    maxpos = hf.getminmaxloc(convolvednormalized)
    print(maxpos)

    crossabspos = hf.getAbsPos(dicomfile.getfilteredarray(), convolvednormalized, maxpos[1])
    dicomfile.setringcoordinates(crossabspos)

    # dicomfile.drawcrosshairs(dicomfile.ringcoordinates)
    ds.imageview3d([dicomfile.getarray(), dicomfile.getfilteredarray()], dicomfile.getname() + "Ring Coordinates")

    dicomfile.setclosestringpoint()
    # dicomfile.drawcrosshairs(dicomfile.closestedge_point)

    hf.cvcontour(dicomfile.getfilteredarray()[dicomfile.closestedge_point[0]])
    hf.blobdetector(dicomfile.getarray()[dicomfile.closestedge_point[0]])
    hf.skiblobdetector(dicomfile.getarray()[dicomfile.closestedge_point[0]])

    ds.imageview3d([dicomfile.getarray(), dicomfile.getfilteredarray()], dicomfile.getname() + "Closest Point")

    # Check to see if ringfocus array was created correctly
    # ds.imageview3d(dicomfile.ringfocusarray(dicomfile.array), "Ringfocus array")

    # Transpose and view rotated array
    dicomfile.createrotatedarrays()
    ds.viewer3d_wRot(dicomfile, "Rotated Array")

    ds.sliceVisualizer(dicomfile, "Slice Visualizer")

    dicomfile.calcsimilaritycoefficient()


# Loop through dicom files and analize them
def analizeExperimentDicom(dicomfileorfolder, excelpath=""):
    if os.path.isdir(dicomfileorfolder):
        for file in ds.getDicomFiles(dicomfileorfolder):
            filesize = os.stat(dicomfileorfolder + '/' + file + ".dcm").st_size
            if filesize > 100000:
                analizeSingleDicom(dicomfileorfolder + '/' + file, excelpath)
    else:
        analizeSingleDicom(dicomfileorfolder, excelpath)
