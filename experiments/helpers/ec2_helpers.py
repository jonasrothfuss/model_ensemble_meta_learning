import datetime
import multiprocessing
import os
import re
import numpy as np
from rllab import config
from scripts.ec2ctl import fetch_zone_prices, fetch_zones


def supported_regions():
    return list( set.intersection(set(config.ALL_REGION_AWS_IMAGE_IDS.keys()),
                            set(config.ALL_REGION_AWS_KEY_NAMES.keys())) )

def spot_history(instance_type, duration='1d'):
    num_duration = int(duration[:-1])
    if re.match(r"^(\d+)d$", duration):
        duration = int(duration[:-1]) * 86400
    elif re.match(r"^(\d+)h$", duration):
        duration = int(duration[:-1]) * 3600
    elif re.match(r"^(\d+)w$", duration):
        duration = int(duration[:-1]) * 86400 * 7
    elif re.match(r"^(\d+)m$", duration):
        duration = int(duration[:-1]) * 86400 * 30
    elif re.match(r"^(\d+)s$", duration):
        duration = int(duration[:-1])
    else:
        raise ValueError("Unrecognized duration: {duration}".format(duration))

    with multiprocessing.Pool(100) as pool:
        zones = sum(pool.starmap(fetch_zones, [(x,) for x in supported_regions()]), [])
        results = pool.starmap(fetch_zone_prices, [(instance_type, zone, duration) for zone in zones])

        price_list = []

        for zone, prices, timestamps in results:
            if len(prices) > 0:
                sorted_prices_ts = sorted(zip(prices, timestamps), key=lambda x: x[1])
                cur_time = datetime.datetime.now(tz=sorted_prices_ts[0][1].tzinfo)
                sorted_prices, sorted_ts = [np.asarray(x) for x in zip(*sorted_prices_ts)]
                cutoff = cur_time - datetime.timedelta(seconds=duration)

                valid_ids = np.where(np.asarray(sorted_ts) > cutoff)[0]
                if len(valid_ids) == 0:
                    first_idx = 0
                else:
                    first_idx = max(0, valid_ids[0] - 1)

                max_price = max(sorted_prices[first_idx:])

                price_list.append((zone, max_price))

    return sorted(price_list, key=lambda x: x[1])

def cheapest_subnets(instance_type, num_subnets=5, duration='1d'):
    spot_prices = spot_history(instance_type, duration=duration)[:num_subnets]

    print("Choosing the following (cheapest) subnets for placing spot requests:")
    for subnet, price in spot_prices:
            print("Subnet: {subnet}, Max Price: {price}".format(subnet=subnet, price=price))

    return [subnet for subnet, price in spot_prices]