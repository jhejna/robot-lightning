import argparse

import yaml
import zerorpc

from robots.controllers.remote import ZeroRPCServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config file.")
    args = parser.parse_args()

    # Then launch the controller.
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    assert config["controller_class"] == "ZeroRPCClient", "Environment must be using ZeroRPCClient."

    controller_class = config["controller_kwargs"]["controller_class"]
    controller_kwargs = config["controller_kwargs"]["controller_kwargs"]

    # Parse the controller and controller kwargs
    server = ZeroRPCServer(controller_class, **controller_kwargs)
    s = zerorpc.Server(server)
    s.bind("tcp://0.0.0.0:4242")
    s.run()
