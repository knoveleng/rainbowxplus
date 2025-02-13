from rainbowxplus.configs.run import ConfigurationLoader


def test_read_config():
    config = ConfigurationLoader.load("./configs/base.yml")
    print(config)


test_read_config()
