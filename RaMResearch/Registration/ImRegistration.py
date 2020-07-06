import SimpleITK as sitk
import math


from OtherPython import registration_gui as rgui, gui


def get_cropped_img(sitk_image, image_dim):
    # Select same subregion using CropImageFilter (NOTE: CropImageFilter cannot reduce dimensions
    # unlike ExtractImageFilter, so cropped_image is a three dimensional image with depth of 1)

    test = sitk_image.GetSize()
    crop_dim = list(map(lambda x: (math.ceil(x/3), math.floor(x*2/3)), test))
    return sitk_image[crop_dim[0][0]:crop_dim[0][1], crop_dim[1][0]:crop_dim[1][1], crop_dim[2][0]:crop_dim[2][1]]
    # crop = sitk.CropImageFilter()
    # crop.SetLowerBoundaryCropSize([100, 100, 0])
    # crop.SetUpperBoundaryCropSize([sitk_image.GetWidth() - 400, sitk_image.GetHeight() - 400, 1])
    # cropped_image = crop.Execute(sitk_image)
    # return cropped_image


def img_registration_sitk(fixed_image, moving_image):

    def resample(image, transform):
        # Output image Origin, Spacing, Size, Direction are taken from the reference
        # image in this call to Resample
        reference_image = image
        interpolator = sitk.sitkLinear
        default_value = 100.0
        return sitk.Resample(image, reference_image, transform,
                             interpolator, default_value)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)

    registration_method.SetOptimizerScalesFromPhysicalShift()
    # Setup for the multi-resolution framework. #registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Don't optimize in-place, we would possibly like to run this cell multiple times .
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    # Connect all of the observers so that we can perform plotting during registratio n.
    registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rgui.plot_values(registration_method))
    final_transform = registration_method.Execute(fixed_image, moving_image)
    # Always check the reason optimization terminated.
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    return resample(moving_image, final_transform)



