from TFL_mng.frame_container import FrameContainer
from TFL_mng.run_attention import test_find_tfl_lights
from TFL_mng.cnn import *
from TFL_mng.SFM import *
class TflMng:
    def __init__(self, pp, focal):
        self.id = None
        self.prev_points = None
        self.frame_container = None
        self.pp = pp
        self.focal = focal
        self.cnn = CNN()
    
    def init_frame(self, image_path, EM):
        self.frame_container = FrameContainer(image_path, EM)

    def run_frame(self):
        self.frame_container.traffic_light = test_find_tfl_lights(self.frame_container.img)
        # plot
        self.frame_container.traffic_light = self.cnn.find_tfl_by_cnn(self.frame_container.img, self.frame_container.traffic_light)
        # plot
        if self.prev_points:
            curr_container = calc_TFL_dist(self.prev_points, self.frame_container, self.focal, self.pp)
            visualize(sequences[i], sequences[i + 1], prev_container, curr_container, focal, pp)
            # plot
        self.prev_points = self.frame_container.traffic_light[:]
        # show

