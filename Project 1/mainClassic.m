clc;
clear;
load("PIController.mat");
Gp = zpk([], [-1 -9], 10);

sysOpenLoop = PIController*Gp;
sys = feedback(PIController*Gp,1);
step(sys);

controlSystemDesigner(Gp,PIController);

save("PIController.mat",'PIController');