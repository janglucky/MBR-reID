from tqdm import tqdm
import time

pbar = tqdm(range(300))
for i in pbar:
	pbar.set_description("Processing %d" % i)
	time.sleep(0.1)

