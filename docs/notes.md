- after every edit play out more steps until an intermediate step is reached to get more signal on reward for that edit 
- use another agent with knowledge about the task and intermediate steps to figure out if the edits were good or not 
- accelerate the convergance by our framework of rewards. reasoning if you use final rewards you can still get convergance 
- potentially how to improve the training infra 
- patterns for edits for example when you get a right intermediate answer remove all steps that lead upto that which are not required 


start with training 1B/2B model on a easy dataset 



new notes 
Mem1 - QA dataset (combine and use all info at once) start with their setup
Be careful of context bench because synthetic 
ICL is a good starting point - good prompt and instructions with auto research 
RL can cause good behaviors in editing to emerge look for these 
1-2B models might not have a good pass rate right out the gate so default to like a 8-9B model as stated in the paper. 