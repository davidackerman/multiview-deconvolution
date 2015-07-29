%details about focus measurements from
%http://vision.fe.uni-lj.si/docs/matejk/MatejKristanPRL06.pdf and from
%QAuto-Pilot paper

function sliceW = estimateSliceSharpnessDCT(slice, blockSize)

%blockSize = 8;

fun = @sharpness;
sliceW = blockproc(slice,[blockSize blockSize],fun, 'UseParallel', true);



end


function Y = sharpness(block_struct)

    t = 6;%cutoff frequency (from Kristan et al). We should determine it experimentally
    
    if( max(block_struct.data(:)) == 0 )
        Y = zeros(size(block_struct.data));
        return;
    end
    
    %calculate dct2 in a block
    Y = dct2(block_struct.data);  
    
    
    %mask things outside cut-off frequency
    [XI, YI] = ndgrid(1:size(Y,1),1:size(Y,2));
    rr = sqrt(XI.^2 + YI.^2);
    Y( rr > t ) = 0;
    
    %normalize DCT
    Y = abs(Y) / norm(Y); %Auto-pilot uses L2 while Kristan et al use L1
    
    %calculate Shannon entropy
    pos = find(Y > 0);
    H = -sum( Y(pos) .* log2(Y(pos)) );
    
    Y = repmat(H,size(Y));
    
end