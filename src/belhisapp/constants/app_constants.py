class AppConstants:
    """ Class with constants used for application visuals, like logos, ASCII art, CSS, etc. """

    LOGO ="""
    ██████╗░███████╗██╗░░░░░██╗░░██╗██╗░██████╗██╗░░██╗░█████╗░░█████╗░██╗
    ██╔══██╗██╔════╝██║░░░░░██║░░██║██║██╔════╝██║░░██║██╔══██╗██╔══██╗██║
    ██████╦╝█████╗░░██║░░░░░███████║██║╚█████╗░███████║███████║███████║██║
    ██╔══██╗██╔══╝░░██║░░░░░██╔══██║██║░╚═══██╗██╔══██║██╔══██║██╔══██║██║
    ██████╦╝███████╗███████╗██║░░██║██║██████╔╝██║░░██║██║░░██║██║░░██║██║
    ╚═════╝░╚══════╝╚══════╝╚═╝░░╚═╝╚═╝╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝╚═╝
    """

    INFO = """
                                                        Welcome to BelHisHAAI V.0.1 Alpha
                                                @basvercruysse @vincentducatteeuw @sanderdiericx
    """

    SHARK = """
                                                ████                                                       
                                                  ███████                    ███████████████               
                                                   █████████        █████████████████████████              
                                                   ███████████ ███████████████████████     █               
                                                    ████████████████████████  ████   █    █                
                                                    ███████████████████████████          ██                
                                                    ██████████████████████              █                  
                                                   ████████ ██ ████████      ███████████                   
                                                  ███████████ ██████     ██ ██ █████ █                     
                                                ██████████  █  ███    ███████████████                      
                                                ███████████ ██ ██    ██████ ██ ███   ██                    
                                               █████████████ █      █████              ██                  
                                              ███████████████         █ ██   ████        ██                
                                              █████  ███████           ██ █      ███████   ██              
                                              ████ ███████                ███                █             
                                              ██ ██████    ████   ████████████   ███           █           
                                                ████    ██        ██████      █                 ██         
                                              ███████  ██       ████           █       ██████              
                                             ████████  █        ██   ███        █ ███        ██            
                                                ██████ █     █████ ███   ███          █████  ██            
                                                  ██████████████████   ███  ███  ████    █████             
                                                    ███████████ ██   ███  ███  █    ████    ███            
                                                       ███████     ███   ██   ████    █████  ██            
                                                         █████         ██   ██   ████    █████             
                                                          █████           ██   █    ████                   
                                                             ███             ██████
    """

    CONFIG_LOGO = """
 ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗ ██╗   ██╗██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝ ██║   ██║██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗██║   ██║██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║██║   ██║██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝╚██████╔╝██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
    """

    CSS = """
        Screen {
            align: center top;
            padding: 1 2;
            background: midnightblue;
        }
        
        HeaderWidget {
            width: 100%;
            height: 20%;
            align: center top;
            content-align: center top;
            background: darkblue;
            border-bottom: heavy steelblue;
        }
        
        HeaderItem {
            color: lightblue;
            text-style: bold;
            text-align: center;
            width: auto;
            margin: 0;
        }
        
        WindowContainer {
            width: 60%;
            background: darkslateblue;
            border: heavy steelblue;
            border-top: thick lightsteelblue;
            border-bottom: thick lightsteelblue;
            content-align: center middle;
            padding: 1 2;
            margin: 1 0;
            
            overflow_y: auto;

            scrollbar-background: midnightblue;
            scrollbar-color: steelblue;
            scrollbar-color-hover: deepskyblue;
            scrollbar-color-active: lightsteelblue;
        }
        
        #default-window-text{
            text-style: bold;
        }
        
        .FormHeader {
           border-bottom: heavy thick lightsteelblue;
           content-align: center middle;
        }
        
        .FormRow {
            height: auto;
            margin: 1 0;
        }
        
        .FormLabel {
            width: 30;
            height: 4;
            text-align: right;
            content-align: right middle;
            padding-right: 2;
            color: lightsteelblue;
            text-style: bold;
        }
        
        .FormTextbox {
            width: 1fr;
            background: midnightblue;
            border: round steelblue;
            color: white;
        }
        
        .FormTextbox:focus {
            border: round deepskyblue;
        }
        
        .FormButton {
            width: 30;
            height: auto;
            background: steelblue;
            color: white;
            border: heavy lightsteelblue;
            text-align: center;
            content-align: center middle;
            text-style: bold;
        }
        
        .FormButton:hover {
            background: deepskyblue;
            color: white;
        }
        
        .FormButton:focus {
            border: heavy deepskyblue;
        }
            
        FooterWidget {
            dock: bottom;
            height: 15%;
            padding: 1 2;
            align: center middle;
            border-top: heavy steelblue;
            background: darkblue;
        }
        
        FooterOption {
            padding: 1 4;
            color: lightblue;
            background: darkblue;
            border: heavy steelblue;
            text-align: center;
            text-style: bold;
            margin: 0 4;
        }
        
        FooterOption.hover {
            background: lightblue;
            color: white;
        }
        
        FooterOption.selected {
            background: deepskyblue;
            color: white;
            text-style: bold;
        }
    """