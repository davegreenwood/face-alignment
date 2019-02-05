import face_alignment
import imageio
import time
import logging
import sys

DEVICE = "cpu" if "c" in sys.argv[1] else "cuda:0"
FNAME = sys.argv[2]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

t0 = time.time()
logger.debug("Module start")
# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                  device=DEVICE,
                                  flip_input=True)
t1 = time.time()
logger.debug("Object Loaded: {:0.3f} seconds".format(t1-t0))

image = imageio.imread(FNAME)
preds = fa.get_landmarks(image)[-1]
t2 = time.time()
logger.debug("First prediction: {:0.3f} seconds".format(t2-t1))

image = imageio.imread(FNAME)
preds = fa.get_landmarks(image)[-1]
t3 = time.time()
logger.debug("Second prediction: {:0.3f} seconds".format(t3-t2))
