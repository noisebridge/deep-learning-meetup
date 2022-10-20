import os
import glob
import cv2
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def gather_data(directory):
    raw_labels=[label for label in os.listdir(directory)]
    imgs = {label: [] for label in raw_labels}
    for label in raw_labels:
        base_path = os.path.join(directory, label)
        image_files = (
                glob.glob(base_path +  "/*.jpeg") +
                glob.glob(base_path +  "/*.png")
        )
        imgs[label] = [cv2.imread(f) for f in image_files]
    return imgs, raw_labels


class LabelMapping:
    def __init__(self, yaml_file, labels):
        self.label_mappings = self._gather_labels(yaml_file)
        self._verify_labels(labels)
        self.output_labels = [
                self.label_mappings[l] for l in labels
        ]

    def _gather_labels(self, yaml_file):
        mappings = {}
        data = {}
        with open(yaml_file) as f:
            data = yaml.load(f, Loader=Loader)
        mappings = data['mappings']
        return mappings

    def _verify_labels(self, labels):
        for l in labels:
            if l not in self.label_mappings:
                return False
        return True


if __name__ == "__main__":
    imgs, raw_labels = gather_data("data")
    mapper = LabelMapping("mappings.yaml", raw_labels)
    print(mapper.output_labels)
