import torchvision


def get_resnet(name):
    resnets = {'resnet18': torchvision.models.resnet18(), 'resnet50': torchvision.models.resnet50()}
    if name not in resnets.keys():
        raise KeyError(f'{name} is not a valid ResNet version')
    return resnets[name]
