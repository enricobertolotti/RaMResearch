

# Ring V2 Class Tester with the following tests:
# 1. Angled Ring Image Loader
# 2. Get cropped image around
def ringv2_test():
    # Import Neccessary Classes
    from RaMResearch.Data import RingV2 as rv2
    from RaMResearch.Utils import Interfaces as intrfce

    # Test loading image
    def loadimage():
        ring_rsmall = 55
        ring_rlarge = 400
        ring_angle = 0
        return rv2.import_ring_array(ring_rsmall, ring_rlarge, ring_angle)

    # Display test image
    def displayimage(ring_image: rv2.RingImage):
        intrfce.imageview3d(ring_image.get_image(), windowName="Ring Image")

    def displaycroppedimage(ring_image):
        intrfce.imageview3d(ring_image, windowName="Cropped Ring Image")

    # Test getting a cropped image around the crosscut
    def getcroppedimage(ring_image: rv2.RingImage):
        return ring_image.get_cropped_image(crop_dim=(400, 400, 400))

    # Run Test
    print("RingV2 Test: Loading Image")
    image = loadimage()
    print("RingV2 Test: Displaying Image")
    # displayimage(image)
    print("RingV2 Test: Getting Cropped Image")
    cropped_image = getcroppedimage(image)
    print("RingV2 Test: Displaying Cropped Image")
    displaycroppedimage(cropped_image)
    print("RingV2 Test: Ring V2 Test Completed.")


# Test Ring Class
ringv2_test()
