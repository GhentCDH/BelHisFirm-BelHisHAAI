import cmd
import os
import os.path
import sys
from index_parser.workflow import Workflow
from index_parser.CRF.train_crf import Train

# Color class to hold ANSI color codes for better readability
class colors:
    INTRO = '\033[1;34m'  # Blue text
    PROMPT = '\033[1;32m'  # Green text for the prompt
    SUCCESS = '\033[1;32m'  # Green for success messages
    ERROR = '\033[1;31m'  # Red for error messages
    RESET = '\033[0m'  # Reset to default color
    WARNING = '\033[1;33m'  # Yellow for warnings

# CLI class
class IndexParserCLI(cmd.Cmd):

    def __init__(self, completekey = "tab", stdin = None, stdout = None):
        super().__init__(completekey, stdin, stdout)

    intro = f'\n{colors.INTRO}You have entered the index parser module. Type help or ? to list commands.\nAlternatively, type --h after a command for more information.\n{colors.RESET}'
    prompt = f'{colors.PROMPT}(IndexParser) {colors.RESET}'

    input_path = None
    model_path = None

    # Command: start
    def do_start(self, arg):
        """Start the IndexParser workflow."""
        if arg == "--h":
            print("Start the IndexParser workflow.")
        else:
            if self.input_path is not None:
                if self.model_path is not None:
                    print(f'{colors.SUCCESS}Starting IndexParser...{colors.RESET}')
                    main_instance = Workflow(self.input_path, self.model_path)
                else:
                    default = input(f'{colors.WARNING}You didn\'t set a model path. Do you want to use the default path? (y/n): {colors.RESET}')
                    if default.lower() == 'y':
                        print(f'{colors.SUCCESS}Starting IndexParser...{colors.RESET}')
                        main_instance = Workflow(self.input_path, self.model_path)

                    else:
                        print(f'{colors.ERROR}Please set a model path path before starting the workflow.{colors.RESET}')
            else:
                default = input(f'{colors.WARNING}You didn\'t set an input path. Do you want to use the default path? (y/n): {colors.RESET}')
                if default.lower() == 'y':
                    print(f'{colors.SUCCESS}Starting IndexParser...{colors.RESET}')
                    main_instance = Workflow(self.input_path, self.model_path)
                    
                else:
                    print(f'{colors.ERROR}Please set an input path before starting the workflow.{colors.RESET}')


    # Command: input
    def do_input(self, arg):
        """Set the input path for the IndexParser workflow"""
        if arg == "--h":
            print("Set the input path for the IndexParser workflow")
        elif arg != "":            
            if os.path.isdir(arg):
                self.input_path = arg
                print(f'{colors.SUCCESS}Input path set to: {colors.RESET}{arg}')
            else:
                print(f'{colors.ERROR}Invalid input path: {colors.RESET}{arg}')
        else:
            print(f'{colors.ERROR}Please enter an input path.{colors.RESET}')

    # Command: model
    def do_model(self, arg):
        """Set the model path for the IndexParser workflow"""
        if arg == "--h":
            print("Set the model path for the IndexParser workflow")
        elif arg != "":            
            if arg.endswith('.pkg'):
                self.model_path = arg
                print(f'{colors.SUCCESS}Model path set to: {colors.RESET}{arg}')
            else:
                print(f'{colors.ERROR}Invalid model path: {colors.RESET}{arg}')
        else:
            print(f'{colors.ERROR}Please enter an model path.{colors.RESET}')
        

    # Command: train
    def do_train(self, arg):
        """Train the model for the IndexParser module"""
        if arg == "--h":
            print(f"{colors.WARNING}Train the model for the IndexParser module\nUSAGE: {colors.RESET}train [CSV_path] [model_name]\n")
        elif arg != "":
            args = arg.split(' ')
            if len(args) < 2:
                print(f"{colors.ERROR}You must provide both [CSV_path] and [model_name] !{colors.RESET}")
                return
            elif args[0].endswith(".csv"):
                trainer = Train()

                path = args[0]
                model_name = args[1]
                c1_value = 0.01
                c2_value = 0.01
                max_iterations_value = 1000

                accuracy = trainer.train(path, model_name, c1_value, c2_value, max_iterations_value)  # Train the model
                print(f"{colors.SUCCESS}[TRAIN]{colors.RESET} Model '{model_name}' trained at {path} | c1={c1_value}, c2={c2_value}, iter={max_iterations_value}, acc={accuracy}")
            else :
                print(f"{colors.ERROR}The [CSV_path] must be a valid CSV file!{colors.RESET}")
        else:
            print(f"{colors.ERROR}This command needs arguments! Use 'train --h' for more info.")

    # Command: return
    def do_return(self, arg):
        """Return to the module selection"""
        if arg == "--h":
            print("Return to the module selection")
        else:
            from cli import BelHisFirmCLI
            indexParserCLI = BelHisFirmCLI()
            indexParserCLI.cmdloop()

    # Command: quit
    def do_quit(self, arg):
        """Quit the CLI."""
        if arg == "--h":
            print("Quit the CLI.")
        else:
            print(f'{colors.ERROR}Closing BelHisFirm CLI...{colors.RESET}')
            sys.exit(0)

    # Keyboard interrupt handling
    def cmdloop(self):
        """Run the command loop with keyboard interrupt handling."""
        try:
            super().cmdloop()
        except KeyboardInterrupt:
            print(f"\n{colors.ERROR}Keyboard interrupt detected. Exiting...{colors.RESET}")
            sys.exit(0)


if __name__ == '__main__':
    IndexParserCLI().cmdloop()
