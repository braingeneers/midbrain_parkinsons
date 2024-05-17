from braingeneers.iot import *                # Import required iot pacakges
import builtins                             # import your own packages here too


def sayHello(name):                         # Write your code into classes or functions
    print("hello ",name)                    
    
class Count:
    count=0
    def add_one(self):
        self.count+=1
    
    
if __name__ == "__main__":                   # Main function called when file is run
    print("starting iot")
    my_count = Count()                       # your starting code (if needed)
    
    ready_iot()                              # Boilerplate code to copy for starting iot. 
    exec( builtins.ready_iot )               # Change: device_name, device_type, experiment
    start_iot( device_name="testit_7", device_type="dummy", experiment="test_experiment" )
    
    print("goodbye")                         # your shutdown code (if needed)


