import urllib.request
import tarfile
import os
import glob


def maybe_download():
    if not os.path.isdir('./datasets/imagenette2-320'):
        print('Downloading ImageNette 320 px')
        thetarfile = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        ftpstream = urllib.request.urlopen(thetarfile)
        thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
        thetarfile.extractall(path='./datasets')


class ImageNette:

    lbl_dict = dict(
        n01440764='tench',
        n02102040='english springer',
        n02979186='cassette player',
        n03000684='chain saw',
        n03028079='church',
        n03394916='french horn',
        n03417042='garbage truck',
        n03425413='gas pump',
        n03445777='golf ball',
        n03888257='parachute'
    )

    def __init__(self):
        maybe_download()
        self.files = glob.glob('./datasets/imagenette2-320/val/*/*')

    def __getitem__(self, index):
        fpath = self.files[index]
        synset = fpath.split('/')[-2]
        target = self.lbl_dict[synset].lower()
        return fpath, target

    def __len__(self):
        return len(self.files)
