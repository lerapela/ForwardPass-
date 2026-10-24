FLAG  				   := -arch=sm_50
NVCC  				   := nvcc
ActivationFunc_Source  := ActivationFunc.cu
MatrixMul_Source       := MatrixMul.cu
Bias_Source 		   := Bias.cu
NNFP_Source			   := NeuralNetForwPass.cu
Target				   := Output

all:
	$(NVCC) $(FLAG) $(ActivationFunc_Source) $(MatrixMul_Source) $(Bias_Source) $(NNFP_Source) -o $(Target)

clean:
	rm -f $(Target)