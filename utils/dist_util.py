import torch.distributed as dist
#This function retrieves the rank (identifier) of the current process in a distributed setup.
def get_rank():
    if not dist.is_available(): #if not dist.is_available():: Checks if the torch.distributed package is available. If not, it returns 0, indicating that the rank is not applicable (typically indicating that no distributed training is occurring).
        return 0
    if not dist.is_initialized():#if not dist.is_initialized():: Checks if the distributed environment is initialized. If it is not initialized, it returns 0, as the rank cannot be determined until the initialization is complete.
        return 0
    return dist.get_rank() # If both checks are passed, it calls dist.get_rank() to get the current process's rank in the distributed group. The rank is a unique identifier for each process (e.g., 0 for the first process, 1 for the second, etc.).

#This function returns the total number of processes in the distributed setup.
def get_world_size():
    if not dist.is_available(): #it checks if torch.distributed is available. If not, it returns 1, indicating that there's only one process (the local process).
        return 1
    if not dist.is_initialized(): #Checks if the distributed environment has been initialized. If not, it returns 1, as the world size cannot be determined in this state.
        return 1
    return dist.get_world_size() #If both checks are passed, it calls dist.get_world_size() to return the total number of processes participating in the distributed training.

# This function checks if the current process is the main process (rank 0).
def is_main_process(): # Calls the get_rank() function and checks if the returned rank is 0. If it is, the function returns True, indicating that this process is the main one. Otherwise, it returns False. The main process often handles tasks like logging and saving models.
    return get_rank() == 0

#: This function formats a string that describes the training and validation steps based on the input step, which can be a string or a list/tuple.
def format_step(step):
    if isinstance(step, str):#: This function formats a string that describes the training and validation steps based on the input step, which can be a string or a list/tuple.
        return step
    s = "" #Initializes an empty string s to build the formatted message.
    if len(step) > 0: # Checks if the step has at least one element. If it does, it appends a string indicating the training epoch to s.
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1: #Checks if the step has at least two elements. If it does, it appends a string indicating the current training iteration.
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2: #Checks if the step has at least three elements. If it does, it appends a string indicating the current validation iteration.
        s += "Validation Iteration: {} ".format(step[2])
    return s

#return s: Finally, it returns the constructed string, which contains information about the current epoch, training iteration, and validation iteration, if applicable.
