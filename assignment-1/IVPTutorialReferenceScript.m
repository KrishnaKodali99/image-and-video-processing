%NAME: FirstName LastName
%INST: IIT, Bhubaneswar
%DATE: 23.08.2017
%CATEGORY: PhD/Btech/Mtech
%BRANCH: Computer Science
%Roll Number: A16EE09005

% Image and Video Tutorial 01
clc;clear;close all;
%%
% Question 1. Convert a color image to grayscale image
% Read an image from Disk
inputImage=imread('peppers.png');
% Check whether it is a color image or not
if size(inputImage,3)==3
    outputImage=rgb2gray(inputImage);
else
    outputImage=inputImage;
end
%Display the original image using a separate figure
figure,imshow(inputImage),title('Input Image ');
%Display the grayscale image using a separate figure
figure,imshow(outputImage),title('Output Image ');
%%
% Question 2. Plot y=x,y=x^2
x=1:0.1:100;
y=x;
figure,plot(x,y,'-ro');
hold on;
y=x.^2;
plot(x,y,'-b+');
grid on;
title('Plot Demo');
xlabel('x');
ylabel('y');
legend('Y=X', 'Y=X^2', 'Location','NorthEast');
hold off;
