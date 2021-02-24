clc;clear;
FLCCarMatlab = readfis('FLCCARMATLAB');
FLCCarSim = FLCCarMatlab;
fuzzyLogicDesigner(FLCCarMatlab);

writefis(FLCCarMatlab,'FLCCARMATLAB');
