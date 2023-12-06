import yaml

class ConfigLoader:
    def __init__(self):
        try:
            with open("./config.yaml") as f:
                self.__config = yaml.full_load(f)
        except:
            self.__config = {}


    def get(self):
        return self.__config