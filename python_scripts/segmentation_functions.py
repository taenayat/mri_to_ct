import SimpleITK as sitk
import vtk
import os
import tqdm
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# def read_dicom(dicom_dir, series = True):
#     reader = sitk.ImageSeriesReader()
    
#     series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)


def read_dicom(dicom_dir):
    reader = sitk.ImageSeriesReader()

    series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)


    tried_serie = 0
    if len(series_IDs) > 1:
        logger.debug("Serveral series detected: {}".format(series_IDs))
        logger.debug("Reading series {} by default".format(series_IDs[-1]))
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[-1])
    else:
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)


    img = None

    ok_read = False
    while not ok_read and tried_serie < len(series_IDs):
        try:
            reader.SetFileNames(dicom_names)
            img = reader.Execute()
            ok_read = True
        except:
            tried_serie += 1
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[tried_serie])


    if img is None:
        raise RuntimeError("No valid series in {}".format(dicom_dir))

    size = img.GetSize()
    
    if len(size) > 3:
        if size[3] == 1:
                size = list(img.GetSize())
                size[3] = 0
                index = [0, 0, 0, 0]
                Extractor = sitk.ExtractImageFilter()
                Extractor.SetSize(size)
                Extractor.SetIndex(index)
                eimg = Extractor.Execute(img)
                img = eimg
        else:
            exit("Error: 4D series")

    return img, reader

def read_strange_dicom(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)

    series_images = {}

    for series in series_IDs:

        print("Reading folder: {}, series: {}".format(dicom_dir, series))

        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series)
        reader.SetFileNames(dicom_names)
        try:
            img = reader.Execute()
        except Exception as e:
            print("[ERROR]: Reading series {} encounter an error".format(series))
            print(e)
            print("[WARNING]: Script will continue with execution skipping this series\n")
            continue
    
        size = img.GetSize()
    
        if len(size) > 3:
            if size[3] == 1:
                    size = list(img.GetSize())
                    size[3] = 0
                    index = [0, 0, 0, 0]
                    Extractor = sitk.ExtractImageFilter()
                    Extractor.SetSize(size)
                    Extractor.SetIndex(index)
                    eimg = Extractor.Execute(img)
                    img = eimg
            else:
                exit("Error: 4D series")

        series_images[series] = img

    return series_images, reader

def get_DICOM_orient(orient):
    x = orient[:3][0]
    y = orient[3:6][1]
    z = orient[6:][2]

    result = ""

    if x < 0:
        result = result + "R"
    else:
        result = result + "L"

    if y < 0:
        result = result + "A"
    else:
        result = result + "P"

    if z < 0:
        result = result + "I"
    else:
        result = result + "S"

    return result

def orient_dicom(img):
    print("itkImage -> GetOrigin() :", img.GetOrigin())
    print("itkImage -> GetDirection() :", img.GetDirection())
    img_orientation = get_DICOM_orient(img.GetDirection())
    print("Orientation: ", img_orientation)
    if img.GetDirection()[8] < 0:
        print("DICOM orientation is {} -> Converting to LPS...".format(img_orientation))
        img = sitk.DICOMOrient(img, 'LPS')
        print("Orientation: ", get_DICOM_orient(img.GetDirection()))

    return img

def crop_dicom(img):
    size = img.GetSize()
    if size[2] > 500:
        crop = sitk.CropImageFilter()
        crop.SetLowerBoundaryCropSize([0,0,size[2]-500])   # This crops the head only
        img = crop.Execute(img)

    return img

def clean_seg(segimg, min_pixels = 500, fullyconn=True):
    ccfilt = sitk.ConnectedComponentImageFilter()
    if fullyconn:
        ccfilt.FullyConnectedOn()
    ccimg = ccfilt.Execute(segimg)

    relab = sitk.RelabelComponentImageFilter()
    relab.SortByObjectSizeOn()
    relab.SetMinimumObjectSize( min_pixels )
    cc_relab = relab.Execute(ccimg)
    
    reseg = cc_relab > 0
    return reseg

def make_clean_mesh(mesh, meshfile, header = None, clean=False, silent=False):
    if not silent:
        print('Cleaning mesh...\n')

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1200, 1200)
    renWin.SetOffScreenRendering(True)
    renWin.AddRenderer(ren)

    ren.AddActor(actor)
    ren.ResetCamera()
    renWin.Render()

    hsel = vtk.vtkHardwareSelector()
    hsel.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
    hsel.SetRenderer(ren)
    hsel.SetArea(0, 0, renWin.GetSize()[0]-1, renWin.GetSize()[1]-1)

    visible_cells_ids = set()

    step = 30
    
    for rot1 in tqdm.tqdm(range(360//step), disable=silent):
        for rot2 in range(360//step):
            renWin.Render()
            res = hsel.Select()
            numNodes = res.GetNumberOfNodes()
            if numNodes > 0:
                sel = res.GetNode(0).GetSelectionList()
                for idx in range(sel.GetNumberOfTuples()):
                    visible_cells_ids.add( int(sel.GetTuple(idx)[0]) )
            ren.GetActiveCamera().Azimuth(step)
        ren.GetActiveCamera().Roll(step)
    
    if not silent:
        print('\n')

    cells = mesh.GetPolys()
    filtered_cells = vtk.vtkCellArray()

    for cellid in range(cells.GetNumberOfCells()):
        if cellid in visible_cells_ids:
            cell_points = vtk.vtkIdList()
            mesh.GetCellPoints( cellid, cell_points )
            filtered_cells.InsertNextCell(cell_points)

    mesh.SetPolys(filtered_cells)

    cleanPolyData = vtk.vtkCleanPolyData()
    
    if clean:
        confilter = vtk.vtkConnectivityFilter()
        confilter.SetInputData(mesh)
        confilter.SetExtractionModeToLargestRegion()
        confilter.Update()
        cleanPolyData.SetInputConnection(confilter.GetOutputPort())
    else:
        cleanPolyData.SetInputData(mesh)

    cleanPolyData.Update()

    mesh1 = cleanPolyData.GetOutput()
    mapper.SetInputData(mesh1)
    renWin.Render()

    writer = vtk.vtkOBJExporter()
    writer.SetFilePrefix(meshfile[:-4])
    if header is not None:
        writer.SetOBJFileComment(header)

    writer.SetInput(renWin)
    writer.Write()

def make_mesh(imgpath, outname, header = None, clean=False, silent=False):
    if imgpath.endswith(".mha"):
        img = vtk.vtkMetaImageReader()
        img.SetFileName(imgpath)
        img.Update()
    elif imgpath.endswith(".nrrd"):
        img = vtk.vtkNrrdReader()
        img.SetFileName(imgpath)
        img.Update()

    elif imgpath.endswith(".nii.gz"):
        sitk_image = sitk.ReadImage(imgpath)
        img = vtk.vtkNIFTIImageReader()
        img.SetFileName(imgpath)
        img.Update()
        matrix = np.eye(4)
        matrix[:3,:3] = np.array(sitk_image.GetDirection()).reshape(3,3)
        matrix[:3, 3] = sitk_image.GetOrigin()

    surface = vtk.vtkMarchingCubes()
    surface.SetInputData(img.GetOutput())
    surface.ComputeNormalsOn()
    surface.SetValue(0, 1.0)
    surface.Update()

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(surface.GetOutput())
    decimate.SetTargetReduction(0.2)
    decimate.Update()
    
    smoothing = vtk.vtkWindowedSincPolyDataFilter()
    smoothing.SetInputConnection(decimate.GetOutputPort())
    smoothing.SetNumberOfIterations(8)
    smoothing.Update()

    normalGenerator = vtk.vtkTriangleMeshPointNormals()
    normalGenerator.SetInputConnection(smoothing.GetOutputPort())
    normalGenerator.Update()

    if imgpath.endswith(".nii.gz"):
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix.flatten())
        transformPolyData = vtk.vtkTransformPolyDataFilter()
        transformPolyData.SetInputData(normalGenerator.GetOutput())
        transformPolyData.SetTransform(transform)
        transformPolyData.Update()
        normalGenerator = transformPolyData

    # objWriter = vtk.vtkPLYWriter()
    # objWriter.SetFileName(outname)
    # objWriter.SetInputData(normalGenerator.GetOutput())
    # objWriter.Write()

    make_clean_mesh(normalGenerator.GetOutput(), outname, header, clean, silent)

def otsu_threshold(image, save_path, scratch):
    threshold_filter = sitk.OtsuMultipleThresholdsImageFilter()
    threshold_filter.SetNumberOfThresholds(2)
    thresh_img = threshold_filter.Execute(image)
    thresholds = threshold_filter.GetThresholds()

    # name = save_path.split(os.sep)[-1]

    # if name == '':
    #     name = save_path.split(os.sep)[-2]

    # if not os.path.exists(os.path.join(scratch, save_path)):
    #     os.makedirs(os.path.join(scratch, save_path))

    # # Save thresholds to txt
    # if not os.path.exists(os.path.join(scratch, save_path, "thresholds_"+name+".txt")):
    #     with open(os.path.join(scratch, save_path, "thresholds_"+name+".txt"), "a") as f:
    #         print(thresholds, file=f)

    return thresh_img, thresholds

def normalize(img):
    filter = sitk.NormalizeImageFilter()
    img = filter.Execute(img)
    array = sitk.GetArrayFromImage(img)
    print(np.min(array), np.max(array))

    return img
    
def segmentation(scratch, tag, threshold, img, th_type):
    fname = os.path.join(scratch, "segmentations", th_type, tag + "_{}_{}.nii.gz".format(th_type, threshold))

    if not os.path.exists(fname):
        print('Segmenting {}...\n'.format(th_type))
        segmentation = img > threshold

        if th_type == "face":
            fill = sitk.GrayscaleFillholeImageFilter()
            fill.FullyConnectedOff()
            seg_fill = fill.Execute(segmentation)
            cast_seg = sitk.Cast(seg_fill, sitk.sitkUInt8 )
        else:
            cast_seg = sitk.Cast(segmentation, sitk.sitkUInt8 )

        cleanseg = clean_seg(cast_seg)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(fname)
        writer.Execute(cleanseg)

    meshname = os.path.join(scratch, "meshes", th_type, tag + "_{}_{}.ply".format(th_type, threshold))

    if not os.path.exists(meshname):
        make_mesh(fname, meshname)


def umbralization(img, thresholds_bone, thresholds_face, tag, scratch):
    for th_bone in thresholds_bone:
        if th_bone is not None:
            segmentation(scratch, tag, th_bone, img, "skull")
    for th_face in thresholds_face:
        if th_face is not None:
            segmentation(scratch, tag, th_face, img, "face")

def segmentation_stats(ref, seg, id_file, filter_list = None):

    if ref.GetPixelIDTypeAsString() != "8-bit unsigned integer":
        logging.info("Converting ref type to UInt8")
        ref = sitk.Cast(ref, sitk.sitkUInt8)

    if seg.GetPixelIDTypeAsString() != "8-bit unsigned integer":
        logging.info("Converting seg type to UInt8")
        seg = sitk.Cast(seg, sitk.sitkUInt8)

    # Overlap measures
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(ref, seg)
    jaccard = overlap_measures_filter.GetJaccardCoefficient()
    dice = overlap_measures_filter.GetDiceCoefficient()
    volume_similarity = overlap_measures_filter.GetVolumeSimilarity()
    false_negative = overlap_measures_filter.GetFalseNegativeError()
    false_positive = overlap_measures_filter.GetFalsePositiveError()

    # Hausdorff distance
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(ref, seg)
    hausdorff_distance = hausdorff_distance_filter.GetHausdorffDistance()

    # Symmetric surface distance measures
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ref, squaredDistance=False, useImageSpacing=True))
    reference_surface = sitk.LabelContour(ref)
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(seg, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(seg)
        
    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map*sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map*sitk.Cast(reference_surface, sitk.sitkFloat32)
        
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    
    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0]) 
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0]) 
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
        
    all_surface_distances = seg2ref_distances + ref2seg_distances

    # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In 
    # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two 
    # segmentations, though in our case it is. More on this below.
    mean_surface_distance = np.mean(all_surface_distances)
    median_surface_distance = np.median(all_surface_distances)
    std_surface_distance = np.std(all_surface_distances)
    max_surface_distance = np.max(all_surface_distances)
    min_surface_distance = np.min(all_surface_distances)
    q1_surface_distance = np.quantile(all_surface_distances, 0.25)
    q2_surface_distance = np.quantile(all_surface_distances, 0.5)
    q3_surface_distance = np.quantile(all_surface_distances, 0.75)
    p99_surface_distance = np.quantile(all_surface_distances, 0.99)

    properties = {}
    properties["JACCARD"] = jaccard
    properties["DICE"] = dice
    properties["VOLUME SIMILARITY"] = volume_similarity
    properties["FALSE NEGATIVE"] = false_negative
    properties["FALSE POSITIVE"] = false_positive
    properties["HAUSDORFF"] = hausdorff_distance
    properties["MEAN SURFACE DISTANCE"] = mean_surface_distance
    properties["MEDIAN SURFACE DISTANCE"] = median_surface_distance
    properties["STD SURFACE DISTANCE"] = std_surface_distance
    properties["MIN SURFACE DISTANCE"] = min_surface_distance
    properties["MAX SURFACE DISTANCE"] = max_surface_distance
    properties["Q1 SURFACE DISTANCE"] = q1_surface_distance
    properties["Q3 SURFACE DISTANCE"] = q3_surface_distance
    properties["P99 SURFACE DISTANCE"] = p99_surface_distance

    properties_row = pd.DataFrame.from_dict({id_file:properties}).T

    if filter_list is not None:
        properties_row = properties_row[filter_list]

    return properties_row