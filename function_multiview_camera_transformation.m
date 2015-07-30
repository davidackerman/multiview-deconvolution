%you can add any transformation needed if you add new microscope
%configurations
function A = function_multiview_camera_transformation(Achoice, anisotropyZ, sizeImRef)

switch(Achoice)
    
    case 10 %reference view: just scale Z        
        A = eye(4);
        A(3,3) = anisotropyZ;
    case 20        
        %original Gcamp6 drosophila experiments: y,z permutation + flip in y
        A = [1 0                0 0;...
             0 0                1 0;...
             0 -anisotropyZ     0 0;...
             0 sizeImRef(1)     0 1];        
    case 21
        %original Gcamp6 zebrafish experiments: y,z permutation + flip in z
        A = [1 0                0               0;...
             0 0                -1              0;...
             0 anisotropyZ      0               0;...
             0 0                sizeImRef(3)    1];
    case 30 %used for opposite camera to 10 (the reference). Just a flip in y       
        A = [1 0            0               0;... 
             0 -1           0               0 ;...
             0 0            anisotropyZ     0;...
             0 sizeImRef(1) 0               1];        
    case 40 %just a flip in z with respect to 20
        %original Gcamp6 drosophila experiments
        A = [1 0            0               0;...
             0 0            -1              0;...
             0 -anisotropyZ 0               0;...
             0 sizeImRef(1) sizeImRef(3)    1];
    case 41 %just a flip in z with respect to 21
        %original Gcamp6 zebrafish experiments
        A = [1 0            0 0; 
             0 0            1 0; 
             0 anisotropyZ  0 0; 
             0 0            0 1];
    otherwise
        error 'Angle not coded here'
        
end