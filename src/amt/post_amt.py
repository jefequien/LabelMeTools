import os
import argparse
import sys
sys.path.append("../coco_utils")
from coco_format import *
from tqdm import tqdm

from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
from boto.mturk.qualification import (Qualifications, 
    PercentAssignmentsApprovedRequirement, 
    NumberHitsApprovedRequirement)
from boto.mturk.price import Price

import config
import hit_properties

class AMTClient:

    def __init__(self, sandbox=True):
        if sandbox:
            self.host = "mechanicalturk.sandbox.amazonaws.com"
            self.external_submit_endpoint = "https://workersandbox.mturk.com/mturk/externalSubmit"
        else:
            self.host = "mechanicalturk.amazonaws.com"
            self.external_submit_endpoint = "https://www.mturk.com/mturk/externalSubmit"

        self.base_url = "https://labelmelite.csail.mit.edu"
        self.connection = MTurkConnection(
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            host=self.host)

    def create_hit(self, job_id, bundle_id, type="edit"):
        params_to_encode = {"job_id": job_id,
                            "bundle_id": bundle_id,
                            "host": self.external_submit_endpoint}
        encoded_url = encode_get_parameters(self.base_url + "/amt_{}".format(type), params_to_encode)
        # print(encoded_url)

        if type == "yesno":
            props = hit_properties.YesNoHitProperties
        elif type == "edit":
            props = hit_properties.EditHitProperties
        else:
            raise Exception("Hit type not implemented")

        create_hit_result = self.connection.create_hit(
            title=props["title"],
            description=props["description"],
            keywords=props["keywords"],
            duration=props["duration"],
            max_assignments=props["max_assignments"],
            question=ExternalQuestion(encoded_url, props["frame_height"]),
            reward=Price(amount=props["reward"]),
            # Determines information returned by certain API methods.
            response_groups=('Minimal', 'HITDetail'),
            qualifications=props["qualifications"])


def encode_get_parameters(baseurl, arg_dict):
    queryString = baseurl + "?"
    for indx, key in enumerate(arg_dict):
        queryString += str(key) + "=" + str(arg_dict[key])
        if indx < len(arg_dict)-1:
            queryString += "&"
    return queryString


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--job_fn', type=str, required=True)
    parser.add_argument('-p', '--prod', action='store_true')
    args = parser.parse_args()
    print(args)

    amt_client = AMTClient(sandbox=(not args.prod))

    job_id = args.job_fn.replace(".txt", "")
    bundle_ids = read_list(args.job_fn)
    for bundle_id in tqdm(bundle_ids):
        amt_client.create_hit(job_id, bundle_id)

