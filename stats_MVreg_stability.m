function [AcellTM, Pcell] = stats_MVreg_stability(filenameXMLpattern, TMvec, plotBool)

%default values as an examples
if( nargin < 1)
  filenameXMLpattern = 'S:\SiMView3\15-07-09\Dre_HuC_H2BGCaMP6s_0-1_20150709_170711.corrected\SPM00\TM??????\MVrefine_deconv_LR_multiGPU_param_TM??????.xml'  
  TMvec = 190:620;
end


%%
%find out number of views
filenameXML = recoverFilenameFromPattern(filenameXMLpattern, TMvec(1));

if( exist(filenameXML,'file') == 0 )
    filenameXML
   error 'Filename XML pattern seems wrong since the file does not exist'
end
Acell = readXMLdeconvolutionFile(filenameXML);

Nviews = length(Acell);
Ntp = length(TMvec);
%%
AcellTM = cell(Nviews, length(TMvec));
for ii = 1:length(TMvec)
    TM = TMvec(ii);
    
    filenameXML = recoverFilenameFromPattern(filenameXMLpattern, TM);
    AcellTM(:,ii) = readXMLdeconvolutionFile(filenameXML);
end
%%
%reorganize output
header = {'TM','Tx', 'Ty', 'Tz', 'a11', 'a12','a13', '211', 'a22','a23', 'a31', 'a32','a33', 'scale_x','scale_y','scale_z','euler_1(deg)','euler_2(deg)','euler_3(deg)'};
Pcell = cell(Nviews,1);
for vv = 1:Nviews
    aux = zeros(Ntp, length(header));
    for ii = 1:Ntp
        [R, T, S] = affineTransformDecomposition(AcellTM{vv,ii});
        Av = AcellTM{vv,ii}(1:3,1:3);
        [e1, e2, e3] = dcm2angle(R);
        aux(ii,:) = [TMvec(ii), T, Av(:)', diag(S)', e1 * 180 /pi, e2 * 180 /pi, e3 * 180 /pi];
    end
    Pcell{vv} = aux;
end
%%
if( plotBool )

    %%
    %plot translations over time for each view
    for vv = 1:Nviews
        xyz = zeros(Ntp,3);
        for ii = 1:Ntp
            xyz(ii,:) = AcellTM{vv,ii}(4,1:3);
        end
        figure;
        plot(TMvec,xyz);
        title(['Translations for view ' num2str(vv)]);
    end
    %%
    
    %%
    %plot coefficients of 3x3 over time for each view
    for vv = 1:Nviews
        xyz = zeros(Ntp,9);
        for ii = 1:Ntp
            qq = AcellTM{vv,ii}(1:3,1:3);
            xyz(ii,:) = qq(:)';
        end
        figure;
        plot(TMvec,xyz);
        title(['Translations for view ' num2str(vv)]);
    end
    
end