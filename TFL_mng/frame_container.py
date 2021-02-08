from PIL import Image

class FrameContainer(object):
    def __init__(self, img_path, EM):
        self.img = Image.open(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = EM
        self.corresponding_ind=[]
        self.valid=[]
