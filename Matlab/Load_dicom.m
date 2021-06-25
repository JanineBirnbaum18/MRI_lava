% Load dicom files from MRI scans

% Signal intensity
path_to_download = 'path\to\diacom\folder';
path_mag = 'BEAT_FQ2';
comp_mag = dir([path_to_download '\' path_mag '\']);
files_mag = dir([path_to_download '\' path_mag '\' comp_mag(3).name]);

path_vel = 'BEAT_FQ2_P';
comp_vel = dir([path_to_download '\' path_vel '\']);
files_vel = dir([path_to_download '\' path_vel '\' comp_vel(3).name]);

close all

figure('units','normalized','outerposition',[0 0 1 1]);
clf;

F(length(files_mag)-2) = struct('cdata',[],'colormap',[]);

for i = 1:(length(files_mag)-2)
    info_mag = dicominfo([path_to_download '\' path_mag '\' comp_mag(3).name '\' files_mag(i+2).name]);
    mag = dicomread(info_mag);
    mag = (double(mag'))/double((2^info_mag.BitDepth));
    subplot(2,1,1)
    imshow(mag,[])
    colorbar parula
    caxis([0 0.075])
    colorbar
    title('Signal Intensity')
    
    info_vel = dicominfo([path_to_download '\' path_vel '\' comp_vel(3).name '\' files_vel(i+2).name]);
    vel = dicomread(info_vel);
    vel = -4*(double(vel')-double(2^(info_vel.BitDepth-1)))/double((2^info_vel.BitDepth-1));
    vel(mag<0.005) = nan;
    subplot(2,1,2)
    imshow(vel,[])
    colorbar parula
    caxis([-1,4])
    colorbar
    title('Velocity (cm/s)')
    
    F(i) = getframe(gcf);
end

v = VideoWriter('output_video_name');
v.FrameRate = 10;
open(v);
writeVideo(v,F);
close(v);
