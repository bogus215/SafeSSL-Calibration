import os

STR2GPU = {     "0" : "MIG-f3304720-4601-5894-bee4-cd0174024e06",
                "1" : "MIG-35d812cd-2b57-5e55-bb30-890bd9675846",
                "2" : "MIG-f166d6fd-d51b-5e97-bbd0-99028db1cd2d",
                "3" : "MIG-d592d9d5-7cd0-5220-8695-37ba249d0084",
                "4" : "MIG-dc45e153-fb1e-5b2d-8a31-d3fb9494cd80",
                "5" : "MIG-21d343f4-de6e-5d44-9774-e2f3dbab968d",
                "6" : "MIG-0b2452d4-9b27-530f-a6f1-1c2d05dfaa72",
                "7" : "MIG-e46a8085-268f-5417-8e5a-a9e20578424d"}

def set_gpu(config):
    if config.server == 'workstation2':
        gpus = ','.join([STR2GPU[str(gpu)] for gpu in config.gpus])
    else:
        gpus = ','.join([str(gpu) for gpu in config.gpus])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus