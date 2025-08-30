
 UPLOADER        David Snyder 
 DATE            2018-05-30
 KALDI VERSION   108832d

 This directory contains files generated from the recipe in
 egs/callhome_diarization/v2/.  It's contents should be placed in a similar
 directory, with symbolic links to diarization/, sid/, steps/, etc.  This was
 created when Kaldi's master branch was at git log
 2ad8d7821867a199e435aa36bbd13af6ed937c94.


 I. Files list
 ------------------------------------------------------------------------------
 
 ./
     README.txt               This file
     run.sh                   A copy of the egs/callhome_diarization/v2/run.sh
                              at the time of uploading this file.  Use this to
                              figure out how to compute features, extract
                              embeddings, etc.

 local/nnet3/xvector/tuning/
     run_xvector_1a.sh        This is the default recipe, at the time of
                              uploading this resource.  The script generates
                              the configs, egs, and trains the model.

 conf/
     vad.conf                 The energy-based VAD configuration
     mfcc.conf                MFCC configuration

 exp/xvector_nnet_1a/ 
     final.raw                The pretrained DNN model
     nnet.config              The nnet3 config file that was used when the
                              DNN model was first instantiated.
     extract.config           Another nnet3 config file that modifies the DNN
                              final.raw to extract x-vectors.  It should be
                              automatically handled by the script 
                              extract_xvectors.sh.
     min_chunk_size           Min chunk size used (see extract_xvectors.sh)
     max_chunk_size           Max chunk size used (see extract_xvectors.sh)
     srand                    The RNG seed used when creating the DNN

 exp/xvectors_callhome1/
     mean.vec                 Vector for centering, from callhome1
     transform.mat            Whitening matrix, trained on callhome1
     plda                     PLDA model for callhome1, trained on SRE data

 exp/xvectors_callhome2/
     mean.vec                 Vector for centering, from callhome2
     transform.mat            Whitening matrix, trained on callhome2
     plda                     PLDA model for callhome1, trained on SRE data


 II. Citation
 ------------------------------------------------------------------------------

 If you wish to use this architecture in a publication, please cite one of the
 following papers.

 The x-vector architecture:

 @inproceedings{snyder2018xvector,
 title={X-vectors: Robust DNN Embeddings for Speaker Recognition},
 author={Snyder, D. and Garcia-Romero, D. and Sell, G. and Povey, D. and Khudanpur, S.},
 booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 year={2018},
 organization={IEEE},
 url={http://www.danielpovey.com/files/2018_icassp_xvectors.pdf}
 }

 Diarization with x-vectors:

 @article{sell2018dihard,
  title={Diarization is Hard: Some Experiences and Lessons Learned for the JHU Team in the Inaugural DIHARD Challenge},
  author={Sell, G. and Snyder, D. and McCree, A. and Garcia-Romero, D. and Villalba, J. and Maciejewski, M. and Manohar, V. and Dehak, N. and Povey, D. and Watanabe, S. and Khudanpur, J.},
  journal={Interspeech},
  year={2018}
 }
 

 III. Recipe README.txt
 ------------------------------------------------------------------------------
 The following text is the README.txt from egs/callhome_diarization/v2 at the
 time this archive was created.

 This recipe replaces i-vectors used in the v1 recipe with embeddings extracted
 from a deep neural network.  In the scripts, we refer to these embeddings as
 "x-vectors."  The x-vector recipe in
 local/nnet3/xvector/tuning/run_xvector_1a.sh is closesly based on the
 following paper: 

 However, in this example, the x-vectors are used for diarization, rather
 than speaker recognition.  Diarization is performed by splitting speech
 segments into very short segments (e.g., 1.5 seconds), extracting embeddings
 from the segments, and clustering them to obtain speaker labels.

 The recipe uses the following data for system development.  This is in
 addition to the NIST SRE 2000 dataset (Callhome) which is used for
 evaluation (see ../README.txt).

     Corpus              LDC Catalog No.
     SRE2004             LDC2006S44
     SRE2005 Train       LDC2011S01
     SRE2005 Test        LDC2011S04
     SRE2006 Train       LDC2011S09
     SRE2006 Test 1      LDC2011S10
     SRE2006 Test 2      LDC2012S01
     SRE2008 Train       LDC2011S05
     SRE2008 Test        LDC2011S08
     SWBD2 Phase 2       LDC99S79
     SWBD2 Phase 3       LDC2002S06
     SWBD Cellular 1     LDC2001S13
     SWBD Cellular 2     LDC2004S07

 The following datasets are used in data augmentation.

     MUSAN               http://www.openslr.org/17
     RIR_NOISES          http://www.openslr.org/28
