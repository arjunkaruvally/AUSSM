# main.py
import sys
from lightning.pytorch.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # parser.link_arguments("data", "model.init_args.data_module", apply_on="instantiate")
        pass


def cli_main():
    cli = MyLightningCLI(parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == "__main__":
    print(" ".join(sys.argv))
    cli_main()
