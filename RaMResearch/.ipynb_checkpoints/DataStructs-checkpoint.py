
class DicomObject:

    # Basic Information
    name = ""
    patient = None
    
    # Metadata for both images
    metadata_no_ring = None
    metadata_with_ring = None

    # Original Images
    image_no_ring = None
    image_with_ring = None

    # Processed Images
    image_difference = None

    def __init__(self, name):
        self.name = name

    def get_image(self, imagetype: str):
        if "without" in imagetype:
            return self.image_no_ring
        if "with" in imagetype:
            return self.image_with_ring
        if "diff" in imagetype:
            return self.image_difference

    def get_name(self):
        return self.name

    def get_patient(self):
        return self.patient

    def get_metadata(self, ring_present: bool):
        return self.metadata

    def set_image(self, image, ring_present: bool):
        if ring_present:
            self.image_with_ring = image.copy()
        else:
            self.image_no_ring = image.copy()

    def set_name(self, name):
        self.name = name
        
    def set_metadata(self, metadata, ring_present: bool):
        self.metadata = metadata
        


