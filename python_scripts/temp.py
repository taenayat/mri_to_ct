print('we entered the file')

import numpy as np # type: ignore
import logging
import time

logging.basicConfig(filename='temp_log.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info('Started')
logger.debug("debog log")
logger.warning('the first warning')

start = time.time()
array = np.array([3,4,5])
print('array:', array)
logger.info('array:'+str(array))
end = time.time()

print(12/0)

print("Total time: {:.1f}".format(end-start))
logger.log(0,'ended')