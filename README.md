# Using LSA and related techniques to efficently search the emails released in the Hacking Team data dump  #

## Dependencies (in order):
	-) libicu
	-) libxml2
	-) libeigen3
	-) libgfortran (from libgcc)
	-) liblapack (makes libblas to link against libsuperlu)
	-) libblas
	-) libsuperlu
	-) libopenblas
	-) libarpack
	-) libarmadillo

## Need to create the following directories
	-) files_not_mined/
	-) raw_matrices/
	-) u_matrices/
	-) v_matrices/
	-) sigma_matricies/
	-) word_vectors/
	-) low_dimensional_space_representation/isigma_ut/
	-) low_dimensional_space_representation/isigma_vt/

## NOTE: still a work in progress


## Goals:
	-) Links to download the coordinate representation of the sparce matricies for each person (if you dont want to download the emails)
	-) SVD of emails for each person
	-) simple binary that takes input arguement of search terms and outputs documets that are most related by some distance measurment (cos(theta) probably)