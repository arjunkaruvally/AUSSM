# main.py
import sys
from lightning.pytorch.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        pass
        parser.link_arguments("data.vocab_size",
                              "model.init_args.network.init_args.vocab_size", apply_on="instantiate")
        parser.link_arguments("data.num_classes",
                              "model.init_args.network.init_args.output_dim", apply_on="instantiate")

def cli_main():
    cli = MyLightningCLI(parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == "__main__":
    print(" ".join(sys.argv))
    cli_main()
