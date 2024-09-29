# -*- coding: utf-8 -*-

"""
@author: Chenfei
@contact:chenfei.ye@foxmail.com
@version: 4.3
@file: run.py
@time: 2023/09/29
# update: support multi session
# update: for multiple b-value dwi, try run eddy seperately, then concatenate together
# update: MNI space res = 1.25 mm, T1w space res = 2 mm
"""


import os, sys, shutil, time, subprocess, inspect, math, glob
import argparse
import numpy as np
import json
import nibabel as nib
import rotate_bvecs_func
import syn
import bids
from bids import BIDSLayout
from grouping import group_dwi_scans

__version__ = 'v4.0'
colourClear = '\033[0m'
colourConsole = '\033[03;32m'
colourError = '\033[01;31m'
colourExec = '\033[03;36m'
colourWarn = '\033[00;31m'


def collect_data(bids_dir, participant_label, filters=None, bids_validate=True):
    """
    Uses pybids to retrieve the input data for a given participant

    """
    if isinstance(bids_dir, BIDSLayout):
        layout = bids_dir
    else:
        layout = BIDSLayout(str(bids_dir), validate=bids_validate)

    queries = {
        'fmap': {'datatype': 'fmap'},
        'sbref': {'datatype': 'func', 'suffix': 'sbref'},
        'flair': {'datatype': 'anat', 'suffix': 'FLAIR'},
        't2w': {'datatype': 'anat', 'suffix': 'T2w'},
        't1w': {'datatype': 'anat', 'suffix': 'T1w'},
        'roi': {'datatype': 'anat', 'suffix': 'roi'},
        'dwi': {'datatype': 'dwi', 'suffix': 'dwi'}
    }
    bids_filters = filters or {}
    for acq, entities in bids_filters.items():
        queries[acq].update(entities)

    subj_data = {
        dtype: sorted(
            layout.get(
                return_type="file",
                subject=participant_label,
                extension=["nii", "nii.gz"],
                **query,
            )
        )
        for dtype, query in queries.items()
    }

    return subj_data, layout


def makeTempDir():
    import random, string
    global tempDir, workingDir
    if tempDir:
        app_error('Script error: Cannot use multiple temporary directories')

    random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(6))
    tempDir = os.path.join(workingDir, 'tmp-' + random_string) + os.sep
    os.makedirs(tempDir)
    app_console('Generated temporary directory: ' + tempDir)
    with open(os.path.join(tempDir, 'cwd.txt'), 'w') as outfile:
        outfile.write(workingDir + '\n')
    with open(os.path.join(tempDir, 'command.txt'), 'w') as outfile:
        outfile.write(' '.join(sys.argv) + '\n')
    open(os.path.join(tempDir, 'log.txt'), 'w').close()

    
def gotoTempDir():
    global tempDir
    if not tempDir:
        app_error('Script error: No temporary directory location set')
    else:
        app_console('Changing to temporary directory (' + tempDir + ')')
        os.chdir(tempDir)


def app_complete():
    global cleanup, tempDir
    global colourClear, colourConsole, colourWarn
    if cleanup and tempDir:
        app_console('Deleting temporary directory ' + tempDir)
        shutil.rmtree(tempDir)
    elif tempDir:
        if os.path.isfile(os.path.join(tempDir, 'error.txt')):
            with open(os.path.join(tempDir, 'error.txt'), 'r') as errortext:
                sys.stderr.write(os.path.basename(sys.argv[0]) + ': ' + colourWarn + 
                                'Script failed while executing the command: ' + errortext.readline().rstrip() + colourClear + '\n')
                sys.stderr.write(os.path.basename(sys.argv[0]) + ': ' + colourWarn + 
                                'For debugging, inspect contents of temporary directory: ' + tempDir + colourClear + '\n')
        else:
            sys.stderr.write(os.path.basename(sys.argv[0]) + ': ' + colourConsole + 
                                'Contents of temporary directory kept, location: ' + tempDir + colourClear + '\n')
            sys.stderr.flush()


def app_console(text):
    global colourClear, colourConsole
    sys.stderr.write(os.path.basename(sys.argv[0]) + ': ' + colourConsole + text + colourClear + '\n')
    if tempDir:
        with open(os.path.join(tempDir, 'log.txt'), 'a') as outfile:
            outfile.write(text + '\n')


def app_warn(text):
    global colourClear, colourWarn
    sys.stderr.write(os.path.basename(sys.argv[0]) + ': ' + colourWarn + '[WARNING] ' + text + colourClear + '\n')
    if tempDir:
        with open(os.path.join(tempDir, 'log.txt'), 'a') as outfile:
            outfile.write(text + '\n')


def app_error(text):
    global colourClear, colourError, cleanup
    sys.stderr.write(os.path.basename(sys.argv[0]) + ': ' + colourError + '[ERROR] ' + text + colourClear + '\n')
    if tempDir:
        with open(os.path.join(tempDir, 'log.txt'), 'a') as outfile:
            outfile.write(text + '\n')
    cleanup = False
    app_complete()
    sys.exit(1)


def mrinfo(image_path, field):
    command_mrinfo = ['mrinfo', image_path, '-' + field ]
    app_console('Command: \'' + ' '.join(command_mrinfo) + '\' (piping data to local storage)')
    proc = subprocess.Popen(command_mrinfo, stdout=subprocess.PIPE, stderr=None)
    result, dummy_err = proc.communicate()
    result = result.rstrip().decode('utf-8')
    return result

# Extract a phase-encoding scheme from a pre-loaded image header,
#   or from a path to the image
def get_pe(arg): #pylint: disable=unused-variable
    if 'PhaseEncodingDirection' in arg.keyval():
        pe = arg.keyval()['PhaseEncodingDirection']
        return pe
    elif 'PhaseEncodingAxis' in arg.keyval():
        pe = arg.keyval()['PhaseEncodingAxis']
        return pe
    elif 'pe_scheme' in arg.keyval():
        app_console(str(arg.keyval()['pe_scheme']))
        return arg.keyval()['pe_scheme']
    else:
        return None


def command(cmd):
    global _processes, app_verbosity, tempDir, cleanup
    global colourClear, colourError, colourConsole
    _env = os.environ.copy()

    return_stdout = ''
    return_stderr = ''
    sys.stderr.write(colourExec + 'Command:' + colourClear + '  ' + cmd + '\n')
    sys.stderr.flush()

    # process = subprocess.Popen(cmd, env=_env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process = subprocess.run(cmd, env=_env, shell=True)
    # return_stdout += process.stdout.readline()

    if process.returncode:
        cleanup = False
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        script_name = os.path.basename(sys.argv[0])
        app_console('')
        try:
            filename = caller.filename
            lineno = caller.lineno
        except AttributeError:
            filename = caller[1]
            lineno = caller[2]
        sys.stderr.write(script_name + ': ' + colourError + '[ERROR] Command failed: ' + cmd + colourClear + colourConsole + ' (' + os.path.basename(filename) + ':' + str(lineno) + ')' + colourClear + '\n')
        sys.stderr.write(script_name + ': ' + colourConsole + 'Output of failed command:' + colourClear + '\n')

        app_console('')
        sys.stderr.flush()
        if tempDir:
            app_complete()
            sys.exit(1)
        else:
            app_warn('Command failed: ' + cmd)

    if tempDir:
        with open(os.path.join(tempDir, 'log.txt'), 'a') as outfile:
            outfile.write(cmd + '\n')

    # return return_stdout, return_stderr


# Class for importing header information from an image file for reading
class Header(object):
    def __init__(self, image_path):
        filename = 'img_header.json'
        command = ['mrinfo', image_path, '-json_all', filename]
        app_console(str(command))
        result = subprocess.call(command, stdout=None, stderr=None)
        if result:
            app_error('Could not access header information for image \'' + image_path + '\'')
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            with open(filename, 'r') as f:
                data = json.loads(f.read().decode('utf-8', errors='replace'))
        os.remove(filename)
        try:
            self._name = data['name']
            self._size = data['size']
            self._spacing = data['spacing']
            self._strides = data['strides']
            self._format = data['format']
            self._datatype = data['datatype']
            self._intensity_offset = data['intensity_offset']
            self._intensity_scale = data['intensity_scale']
            self._transform = data['transform']
            if not 'keyval' in data or not data['keyval']:
                self._keyval = {}
            else:
                self._keyval = data['keyval']
        except:
            app_error('Error in reading header information from file \'' + image_path + '\'')
        app_console(str(vars(self)))
    def name(self):
        return self._name
    def size(self):
        return self._size
    def spacing(self):
        return self._spacing
    def strides(self):
        return self._strides
    def format(self):
        return self._format
    def datatype(self):
        return self.datatype
    def intensity_offset(self):
        return self._intensity_offset
    def intensity_scale(self):
        return self._intensity_scale
    def transform(self):
        return self._transform
    def keyval(self):
        return self._keyval

# Computes image statistics using mrstats.
# Return will be a list of ImageStatistics instances if there is more than one volume
#   and allvolumes=True is not set; a single ImageStatistics instance otherwise
from collections import namedtuple
ImageStatistics = namedtuple('ImageStatistics', 'mean median std std_rv min max count')
IMAGE_STATISTICS = ['mean', 'median', 'std', 'std_rv', 'min', 'max', 'count' ]

def img_statistics(image_path, **kwargs):

    mask = kwargs.pop('mask', None)
    allvolumes = kwargs.pop('allvolumes', False)
    ignorezero = kwargs.pop('ignorezero', False)
    if kwargs:
        raise TypeError('Unsupported keyword arguments passed to image.statistics(): ' + str(kwargs))

    command = ['mrstats', image_path ]
    for stat in IMAGE_STATISTICS:
        command.extend([ '-output', stat ])
    if mask:
        command.extend([ '-mask', mask ])
    if allvolumes:
        command.append('-allvolumes')
    if ignorezero:
        command.append('-ignorezero')

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None)
    stdout = proc.communicate()[0]
    if proc.returncode:
        raise OSError('Error trying to calculate statistics from image \'' + image_path + '\'')
    stdout_lines = [line.strip() for line in stdout.decode('cp437').splitlines() ]
    result = []
    for line in stdout_lines:
        line = line.replace('N/A', 'nan').split()
        assert len(line) == len(IMAGE_STATISTICS)
        result.append(ImageStatistics(float(line[0]), float(line[1]), float(line[2]), float(line[3]),
                                      float(line[4]), float(line[5]), int(line[6])))
    if len(result) == 1:
        result = result[0]
    return result


def get_image_spacing(img_path):
    img = nib.load(img_path)
    # affine = img.affine
    # return str(abs(round(affine[0, 0], 2)))
    return str(abs(np.min(img.header.get_zooms()[0:3])))


# Load a text file containing numeric data
#   (can be a different number of entries in each row)
def load_numeric(filename, **kwargs):
    import re, itertools
    dtype = kwargs.pop('dtype', float)
    # By default support the same set of delimiters at load as the MRtrix3 C++ code
    delimiter = kwargs.pop('delimiter', ' ,;\t')
    comments = kwargs.pop('comments', '#')
    encoding = kwargs.pop('encoding', 'latin1')
    errors = kwargs.pop('errors', 'ignore')
    if kwargs:
        raise TypeError('Unsupported keyword arguments passed to matrix.load_numeric(): ' + str(kwargs))

    def decode(line):
        if isinstance(line, bytes):
            line = line.decode(encoding, errors=errors)
        return line

    if comments:
        regex_comments = re.compile('|'.join(comments))

    data = []
    with open(filename, 'rb') as infile:
        for line in infile.readlines():
            line = decode(line)
        if comments:
            line = regex_comments.split(line, maxsplit=1)[0]
        line = line.strip()
        if line:
            if len(delimiter) == 1:
                data.append([dtype(a) for a in line.split(delimiter) if a])
            else:
                data.append([dtype(a) for a in [''.join(g) for k, g in itertools.groupby(line, lambda c : c in delimiter) if not k ]])

    if not data:
        return None
    return data


# Load a text file containing specifically matrix data
def load_matrix(filename, **kwargs): #pylint: disable=unused-variable
    data = load_numeric(filename, **kwargs)
    columns = len(data[0])
    for line in data[1:]:
        if len(line) != columns:
            raise OSError('Inconsistent number of columns in matrix text file "' + filename + '"')
    return data


# Load a text file containing specifically vector data
def load_vector(filename, **kwargs): #pylint: disable=unused-variable
    data = load_matrix(filename, **kwargs)
    if len(data) == 1:
        return data[0]
    for line in data:
        if len(line) != 1:
            raise OSError('File "' + filename + '" does not contain vector data (multiple columns detected)')
    return [line[0] for line in data]


# dwi mask loop creation
def dwi_mask_loop_create(dwi_image, dwi_mask_image, dwi_validdata_image, multishell):
    global tempDir
    # create a tmp folder
    cwd_dir = os.path.dirname(dwi_image)
    tmp_mask_dir = os.path.join(cwd_dir, 'dwi_mask_loop_tmp')
    if not os.path.exists(tmp_mask_dir):
        os.mkdir(tmp_mask_dir)

    # move input images to tmp folder
    dwi_image_base = os.path.basename(dwi_image)
    dwi_image_indef = os.path.join(tmp_mask_dir, dwi_image_base)

    dwi_mask_image_base = os.path.basename(dwi_mask_image)
    dwi_mask_image_indef = os.path.join(tmp_mask_dir, dwi_mask_image_base)

    dwi_validdata_image_base = os.path.basename(dwi_validdata_image)
    dwi_validdata_image_indef = os.path.join(tmp_mask_dir, dwi_validdata_image_base)

    shutil.copyfile(dwi_image, dwi_image_indef)
    shutil.copyfile(dwi_mask_image, dwi_mask_image_indef)
    shutil.copyfile(dwi_validdata_image, dwi_validdata_image_indef)

    # go into tmp folder
    os.chdir(tmp_mask_dir)


    # Combined RF estimation / CSD / mtnormalise / mask revision
    class Tissue(object):
        def __init__(self, name, index):
            self.name = name
            iter_string = '_iter' + str(index)
            self.rf = 'response_' + name + iter_string + '.txt'
            self.fod_init = 'FODinit_' + name + iter_string + '.mif'
            self.fod_norm = 'FODnorm_' + name + iter_string + '.mif'

    app_console('Commencing iterative DWI bias field correction and brain masking')

    DWIBIASCORRECT_MAX_ITERS = 2
    TISSUESUM_THRESHOLD = 0.5 / math.sqrt(4.0 * math.pi)


    for iteration in range(0, DWIBIASCORRECT_MAX_ITERS):
        iter_string = '_iter' + str(iteration + 1)

        tissues = [Tissue('WM', iteration),
                   Tissue('GM', iteration),
                   Tissue('CSF', iteration)]

        step = 'dwi2response'
        command('dwi2response dhollander ' + dwi_image_base + ' -mask ' + dwi_mask_image_base + ' '
                + ' '.join(tissue.rf for tissue in tissues))

        # Remove GM if we can't deal with it
        lmaxes = '4,0,0'
        if not multishell:
            tissues = tissues[::2]
            lmaxes = '4,0'

        step = 'dwi2fod'
        command('dwi2fod msmt_csd ' + dwi_image_base + ' -lmax ' + lmaxes + ' ' +
                ' '.join(tissue.rf + ' ' + tissue.fod_init for tissue in tissues))

        step = 'mtnormalise'
        field_path = 'field' + iter_string + '.mif'
        factors_path = 'factors' + iter_string + '.txt'
        command('maskfilter ' + dwi_mask_image_base + ' erode - |' + ' mtnormalise -mask - -balanced'
                + ' -check_norm ' + field_path + ' -check_factors ' + factors_path
                + ' ' + ' '.join(tissue.fod_init + ' ' + tissue.fod_norm for tissue in tissues))

        # Apply both estimated bias field, and appropiate
        #   scaling factor, to DWIs
        step = 'mrcalc_dwi'
        csf_rf = load_matrix(tissues[-1].rf)
        csf_rf_bzero_lzero = csf_rf[0][0]
        balance_factors = load_vector(factors_path)
        csf_balance_factor = balance_factors[-1]
        scale_multiplier = (1000.0 * math.sqrt(4.0 * math.pi)) / \
                           (csf_rf_bzero_lzero / csf_balance_factor)
        new_dwi_image_base = 'dwi' + iter_string + '.mif'
        command('mrcalc ' + dwi_image_base + ' '
                + field_path + ' -div '
                + str(scale_multiplier) + ' -mult '
                + new_dwi_image_base)
        dwi_image_base = new_dwi_image_base

        step = 'dwi2mask'
        new_dwi_mask_image_base = 'dwi_mask' + iter_string + '.mif'
        command('mrconvert '
                + tissues[0].fod_norm
                + ' -coord 3 0 - |'
                + ' mrmath - '
                + ' '.join(tissue.fod_norm for tissue in tissues[1:])
                + ' sum - |'
                + ' mrthreshold - -abs '
                + str(TISSUESUM_THRESHOLD)
                + ' - |'
                + ' maskfilter - connect -largest - |'
                + ' mrcalc 1 - -sub - -datatype bit |'
                + ' maskfilter - connect -largest - |'
                + ' mrcalc 1 - -sub - -datatype bit |'
                + ' maskfilter - clean - |'
                + ' mrcalc - '
                + dwi_validdata_image_base
                + ' -mult '
                + new_dwi_mask_image_base
                + ' -datatype bit')

        # Compare input and output masks
        step = 'mrcalc_mask'
        dwi_old_mask_count = img_statistics(dwi_mask_image_base,
                                            mask=dwi_mask_image_base).count
        dwi_new_mask_count = img_statistics(new_dwi_mask_image_base,
                                            mask=new_dwi_mask_image_base).count
        app_console('Old mask: ' + str(dwi_old_mask_count))
        app_console('New mask: ' + str(dwi_new_mask_count))
        dwi_mask_overlap_image_base = 'dwi_mask_overlap' + iter_string + '.mif'
        command('mrcalc ' + dwi_mask_image_base + ' ' + new_dwi_mask_image_base + ' -mult ' + dwi_mask_overlap_image_base)

        dwi_mask_image_base = new_dwi_mask_image_base
        mask_overlap_count = img_statistics(dwi_mask_overlap_image_base, mask=dwi_mask_overlap_image_base).count
        app_console('Mask overlap: ' + str(mask_overlap_count))
        dice_coefficient = 2.0 * mask_overlap_count / \
                           (dwi_old_mask_count + dwi_new_mask_count)
        app_console('Dice coefficient: ' + str(dice_coefficient))
        if dice_coefficient > (1.0 - 1e-3):
            app_console('Exiting iterative loop due to mask convergence')
            break

    # return to parent folder
    os.chdir(tempDir)
    dwi_mask_image_output = os.path.join(cwd_dir, 'dwi_mask_image.mif')
    shutil.copyfile(os.path.join(tmp_mask_dir, dwi_mask_image_base), dwi_mask_image_output)

    return dwi_mask_image_output



def preprocess_dwi(args, dwi_preproc_folder, t1_dir, dwi_image_input, dwi_image_output, dwi_image_json, dwi_image_bvec, dwi_image_bval, step1_t1_2_mni_reverse_transform):
    # denoise, gibbs ringing removal and eddy
    dwi_gibbs_bzero_path = os.path.join(tempDir, dwi_preproc_folder, 'dwi_gibbs_bzero.nii.gz')
    dwi_gibbs_sdc_bzero_path = os.path.join(tempDir, dwi_preproc_folder, 'dwi_gibbs_sdc_bzero.nii.gz')
    dwipreproc_input_nii = os.path.join(dwi_preproc_folder, 'dwi_gibbs.nii.gz')
    dwipreproc_sdc_nii = os.path.join(dwi_preproc_folder, 'dwi_gibbs_sdc.nii.gz')
    dwipreproc_sdc_mif = os.path.join(dwi_preproc_folder, 'dwi_gibbs_sdc.mif')
    dwi_denoise_output = os.path.join(dwi_preproc_folder, 'dwi_denoised.mif')
    dwipreproc_input = os.path.join(dwi_preproc_folder, 'dwi_gibbs.mif')
    step1_t1_ss_path = os.path.join(t1_dir, 'T1w_proc_ss.nii.gz')


    # dwidenoise
    app_console('Performing dwi data denoise')
    command('dwidenoise ' + dwi_image_input + ' ' + dwi_denoise_output)

    # Gibbs ringing removal
    app_console('Performing Gibbs ringing removal for DWI data')
    command('mrdegibbs ' + dwi_denoise_output + ' ' + dwipreproc_input + ' -nshifts 50')

    # read dwi_pe
    dwipreproc_input_header = Header(dwipreproc_input)
    dwi_pe = get_pe(dwipreproc_input_header)

    app_console('T1prep folder detected, will run fieldmap-less susceptibility distortion correction using ANTs')
    command('dwiextract ' + dwipreproc_input + ' -bzero - | '
        'mrcalc - 0.0 -max - | '
        'mrmath - mean -axis 3 ' + dwi_gibbs_bzero_path + ' -force')
    b0_ref = dwi_gibbs_bzero_path
    syn_sdc_wf = syn.init_syn_sdc_wf(step1_t1_ss_path, step1_t1_2_mni_reverse_transform, b0_ref, omp_nthreads=8, bold_pe=dwi_pe, atlas_threshold=2, name='syn_sdc_wf') 
    syn_sdc_wf.base_dir = dwi_preproc_folder
    syn_sdc_wf.run()
    app_console('finished fieldmap-less susceptibility distortion correction using ANTs, now map warpping on raw 4D DWI')
    app_console('This method is modified from QSIPrep: https://github.com/PennLINC/qsiprep')

    command('mrconvert ' + dwipreproc_input + ' ' + dwipreproc_input_nii)
    command('antsApplyTransforms -d 3 -e 3  -i ' + dwipreproc_input_nii + ' -r ' + 
            os.path.join(dwi_preproc_folder, 'syn_sdc_wf/unwarp_ref/dwi_gibbs_bzero_trans.nii.gz') + ' -t ' + 
            os.path.join(dwi_preproc_folder, 'syn_sdc_wf/syn/ants_susceptibility0Warp.nii.gz') + 
            ' -o ' + dwipreproc_sdc_nii + ' -v --float -n LanczosWindowedSinc -f 0')
    shutil.copyfile(os.path.join(dwi_preproc_folder, 'syn_sdc_wf/unwarp_ref/dwi_gibbs_bzero_trans.nii.gz'), dwi_gibbs_sdc_bzero_path)
    # Note that 'syn_sdc_wf/syn/ants_susceptibility0Warp.nii.gz' is actually 5D image with shape = 110 x 110 x 68 x 1 x 3
    command('mrconvert ' + dwipreproc_sdc_nii + ' ' + dwipreproc_sdc_mif + ' -json_import ' +
        dwi_image_json + ' -fslgrad ' + dwi_image_bvec + ' ' + dwi_image_bval)


    # for fast preprocessing mode, just denoise/gibbs dwi; for complete preprocessing mode, perform denoise/gibbs/eddy_topup
    # Distortion correction
    app_console('Performing topup and eddy correction')
    dwipreproc_input_header = Header(dwipreproc_input)
    mb_factor = int(dwipreproc_input_header.keyval().get('MultibandAccelerationFactor', '1'))

    if 'SliceDirection' in dwipreproc_input_header.keyval():
        slice_direction_code = dwipreproc_input_header.keyval()['SliceDirection']
        if 'i' in slice_direction_code:
            num_slices = dwipreproc_input_header.size()[0]
        elif 'j' in slice_direction_code:
            num_slices = dwipreproc_input_header.size()[1]
        elif 'k' in slice_direction_code:
            num_slices = dwipreproc_input_header.size()[2]
        else:
            num_slices = dwipreproc_input_header.size()[2]
            app_warn('Error reading BIDS field \'SliceDirection\' (value: \'' + slice_direction_code + '\'); assuming third axis')
    else:
        num_slices = dwipreproc_input_header.size()[2]
    mporder = 1 + num_slices / (mb_factor * 4)
    dwipreproc_eddy_option = ''
    
    if not os.path.exists(os.path.join(tempDir, 'eddyqc')):
        os.mkdir(os.path.join(tempDir, 'eddyqc'))
    
    dwi_pe = get_pe(dwipreproc_input_header)
    if not dwi_pe:
        app_error('No phase encoding information found in DWI image header')

    if args.mode == 'complete':
        app_console('run eddy correction')
        if not os.path.exists(dwi_image_output):
            command('dwifslpreproc ' + dwipreproc_sdc_mif + ' -fslgrad ' + dwi_image_bvec + ' ' + dwi_image_bval + ' ' + 
                    dwi_image_output + ' -rpe_none -pe_dir ' + dwi_pe + ' ' + dwipreproc_eddy_option + ' -eddyqc_text eddyqc/ -force')
    else:
        shutil.copyfile(dwipreproc_sdc_mif, dwi_image_output)
    
    # remove temporary files
    if os.path.exists(dwipreproc_input_nii):
        os.remove(dwipreproc_input_nii)

    if os.path.exists(dwipreproc_input_nii):
        os.remove(dwipreproc_input_nii)

    if os.path.exists(dwipreproc_sdc_nii):
        os.remove(dwipreproc_sdc_nii)
    
    if os.path.exists(dwipreproc_sdc_mif):
        os.remove(dwipreproc_sdc_mif)
    
    if os.path.exists(dwi_denoise_output):
        os.remove(dwi_denoise_output)
    
    if os.path.exists(dwipreproc_input):
        os.remove(dwipreproc_input)

    app_console('complete dwi preprocessing')
    app_console('-------------------------------------')



def runSubject(args, subject_label, session_label, t1prep_dir):
    global workingDir, tempDir, cleanup, resume
    label = 'sub-' + subject_label

    if session_label:
        t1_dir = os.path.join(t1prep_dir, label,  'ses-' + session_label)
        output_dir = os.path.join(args.output_dir, label,  'ses-' + session_label)
        app_console('Launching participant-level analysis for subject \'' + label + '\'' + ' and session \'' + session_label + '\'')
    else:
        t1_dir = os.path.join(t1prep_dir, label)
        output_dir = os.path.join(args.output_dir, label)
        app_console('Launching participant-level analysis for subject \'' + label + '\'')
    
    app_console('output_dir:'+ output_dir)
    if os.path.exists(output_dir):
        app_warn('Output directory for subject \'' + label + '\' already exists. would erase the original folder by default')
        shutil.rmtree(output_dir)
    
    if not os.path.exists(t1_dir):
        app_console('t1_dir: '+ t1_dir)
        app_error('Failed to detect output folder of BIDS-T1prep for subject ' + label)
    
    # if -resume, find the last modified temp folder
    if resume:
        tmp_folder_find = glob.glob(os.path.join(workingDir, 'tmp-*'))
        if len(tmp_folder_find) == 0:
            app_warn('Found no tmp folder in resume mode, create a new tmp folder')
            makeTempDir()
            gotoTempDir()
        else:
            modified_time_ls = [os.path.getmtime(path) for path in tmp_folder_find]
            max_idx = modified_time_ls.index(max(modified_time_ls))
            tempDir = tmp_folder_find[max_idx]
            os.chdir(tempDir)
    else:
        makeTempDir()
        gotoTempDir()
    app_console('working directory: ' + os.getcwd())

    app_console('Launching participant-level analysis for subject \'' + label + '\'')

    # initialize input folder
    dwi_input_folder = 'preproc_dwi_input'
    dwi_image = os.path.join(dwi_input_folder, 'dwi.mif') # input dwi image
    # dwi_image_nii = os.path.join(dwi_input_folder, 'dwi.nii.gz') # input dwi image
    dwi_image_bvec = os.path.join(dwi_input_folder, 'dwi.bvec') # input dwi image
    dwi_image_bval = os.path.join(dwi_input_folder, 'dwi.bval') # input dwi image
    dwi_image_json = os.path.join(dwi_input_folder, 'dwi.json') # input dwi image


    layout = bids.layout.BIDSLayout(args.bids_dir, derivatives=False, absolute_paths=True)
    dwi_image_list = [f.path for f in layout.get(subject=subject_label,suffix='dwi',extension=["nii.gz", "nii"],**session_to_analyze)]
    dwi_bval_list = [f.path for f in layout.get(subject=subject_label,suffix='dwi',extension=["bval"],**session_to_analyze)]
    dwi_bvec_list = [f.path for f in layout.get(subject=subject_label,suffix='dwi',extension=["bvec"],**session_to_analyze)]
    dwi_json_list = [f.path for f in layout.get(subject=subject_label,suffix='dwi',extension=["json"],**session_to_analyze)]

    # subject_data, layout = collect_data(args.bids_dir, subject_label)
    # dwi_fmap_groups, concatenation_scheme = group_dwi_scans(
    #     layout, subject_data,
    #     using_fsl=True,
    #     combine_scans=True,
    #     concatenate_distortion_groups=False)
    
    if os.path.exists(dwi_image):
        app_console('Step 0 output found, skip this step')
    else:
        if not os.path.exists(os.path.join(tempDir, dwi_input_folder)):
            os.mkdir(os.path.join(tempDir, dwi_input_folder))
        # Import diffusion image
        app_console('Importing diffusion image into temporary directory')

        if len(dwi_image_list) > 1:
            app_console('multiple dwi found, will try to concatenate all dwi images')
            dwi_image_mif_ls = []
            for item in range(len(dwi_image_list)):
                app_console('dwi found: ' + dwi_image_list[item])
                dwi_image_mif = os.path.join(dwi_input_folder, os.path.basename(dwi_image_list[item]).split('.')[0] + '.mif')
                command('mrconvert ' + dwi_image_list[item] + ' ' + dwi_image_mif + ' -json_import ' +
                    dwi_json_list[item] + ' -fslgrad ' + dwi_bvec_list[item] + ' ' + dwi_bval_list[item] + ' -strides -1,2,3,4')
                dwi_image_mif_ls.append(dwi_image_mif)
            # command('mrcat ' + ' '.join(dwi_image_mif_ls) + ' ' + dwi_image )
            # command('mrconvert ' + dwi_image + ' ' + dwi_image_nii + ' -export_grad_fsl ' + dwi_image_bvec + ' ' + dwi_image_bval + ' -json_export ' + dwi_image_json)

        elif not dwi_image_list:
            app_error('No dwi image found for subject ' + label)
        else:
            dwi_image_path = dwi_image_list[0]
            app_console('one dwi found: ' + dwi_image_path)
            try:
                command('mrconvert ' + dwi_image_path + ' ' + dwi_image + ' -json_import ' +
                        dwi_json_list[0] + ' -fslgrad ' + dwi_bvec_list[0] + ' ' + dwi_bval_list[0] + ' -strides -1,2,3,4')
                shutil.copyfile(dwi_bvec_list[0], dwi_image_bvec)
                shutil.copyfile(dwi_bval_list[0], dwi_image_bval)
                shutil.copyfile(dwi_json_list[0], dwi_image_json)
            except:
                app_error('one of dwi.nii.gz/dwi.json/dwi.bval/dwi.bvec files is lacking, or failed to convert these files into mif format')

    qc_dir = os.path.join(tempDir, 'qc')
    if not os.path.exists(qc_dir):
        os.mkdir(qc_dir)
    
    #-----------------------------------------------------------------
    # Step 1: dwi round1-preprocessing
    #-----------------------------------------------------------------
    dwi_preproc_folder = 'preproc_dwi_proc'

    # initialize output folder and file
    if not os.path.exists(os.path.join(tempDir, dwi_preproc_folder)):
        os.mkdir(os.path.join(tempDir, dwi_preproc_folder))
    step1_output_dwi_image = os.path.join(dwi_preproc_folder, 'dwi_preproc.mif')
    dwi_gibbs_bzero_path = os.path.join(tempDir, dwi_preproc_folder, 'dwi_gibbs_bzero.nii.gz')


    step1_t1_path = os.path.join(t1_dir, 'T1w_proc.nii.gz')
    step1_t1_mask_path = os.path.join(t1_dir, 'T1w_bet_mask.nii.gz')
    step1_t1_ss_path = os.path.join(t1_dir, 'T1w_proc_ss.nii.gz')
    step1_t1_2_mni_reverse_transform = os.path.join(t1_dir, 'composite_warp_mni_to_t1.nii.gz')
    dwi_to_t1_affine = os.path.join(dwi_preproc_folder, 'syn_sdc_wf/ref_2_t1/transform0GenericAffine.mat')


    if args.resume and os.path.exists(step1_output_dwi_image): # if -resume, check if output file exist or not
        app_console('Step 1 output dwipreproc_sdc_mif found, skip this step')
    else:  # no output file found, begin process
        app_console('run Step 1: dwi round1-preprocessing')
        if args.mode == 'fast':
            app_console('choose fast preprocessing')
        elif args.mode == 'complete':
            app_console('choose complete preprocessing')
        
        if len(dwi_image_list) > 1:
            dwi_image_output_ls = [] 
            # preprocess for each dwi
            for item in range(len(dwi_image_list)):
                dwi_image_output = os.path.join(dwi_input_folder, os.path.basename(dwi_image_list[item]).split('.')[0] + '_eddy.mif')
                preprocess_dwi(args, dwi_preproc_folder, t1_dir, dwi_image_mif_ls[item], \
                        dwi_image_output, dwi_json_list[item], dwi_bvec_list[item], \
                        dwi_bval_list[item], step1_t1_2_mni_reverse_transform)
                dwi_image_output_ls.append(dwi_image_output)
            # concatenate together
            command('mrcat ' + ' '.join(dwi_image_output_ls) + ' ' + step1_output_dwi_image )
        else:
            dwi_image_output = os.path.join(dwi_input_folder, os.path.basename(dwi_image_list[0]).split('.')[0] + '_eddy.mif')
            preprocess_dwi(args, dwi_preproc_folder, t1_dir, dwi_image, \
                        dwi_image_output, dwi_image_json, dwi_image_bvec, \
                        dwi_image_bval, step1_t1_2_mni_reverse_transform)
            step1_output_dwi_image = dwi_image_output
            

    #-----------------------------------------------------------------
    # Step 2: dwi round2-preprocessing: b0 extraction and dwimask creation
    #-----------------------------------------------------------------
    step2_output_dwi_final_mif = os.path.join(dwi_preproc_folder, 'Diffusion_iso.mif')
    step2_output_dwi_final_nii = os.path.join(dwi_preproc_folder, 'Diffusion_iso.nii.gz')
    step2_output_dwi_final_bvec = os.path.join(dwi_preproc_folder, 'Diffusion_iso.bvec')
    step2_output_dwi_final_bval = os.path.join(dwi_preproc_folder, 'Diffusion_iso.bval')
    step2_output_dwi_mask_final_nii = os.path.join(dwi_preproc_folder, 'Diffusion_iso_mask.nii.gz')
    step2_output_b0_final_nii = os.path.join(dwi_preproc_folder, 'Diffusion_b0_iso.nii.gz')
    
    step2_output_dwi_proc_nii = os.path.join(dwi_preproc_folder, 'Diffusion_raw.nii.gz')
    step2_output_dwi_mask_image = os.path.join(dwi_preproc_folder, 'Diffusion_mask.nii.gz')
    step2_output_dwi_bvec = os.path.join(dwi_preproc_folder, 'Diffusion.bvec')
    step2_output_dwi_bval = os.path.join(dwi_preproc_folder, 'Diffusion.bval')
    step2_output_bzero = os.path.join(dwi_preproc_folder, 'Diffusion_b0.nii.gz')
    
    dwi_t1_FSL_mat = os.path.join(dwi_preproc_folder, 'dwi_t1_FSL.mat')
    step2_t1_2mm_ss_path = os.path.join(dwi_preproc_folder, 'T1w_2mm_ss.nii.gz')
    step2_t1_2mm_mask_path = os.path.join(dwi_preproc_folder, 'T1w_2mm_mask.nii.gz')

    MNI_T2w_template = '/data/tpl-MNI152NLin2009cAsym_space-MNI_res-0125_T2w.nii.gz'
    MNI_T2w_template_mask = '/data/tpl-MNI152NLin2009cAsym_space-MNI_res-0125_brainmask.nii.gz'
    MNI_lin_ANTs_mat = os.path.join(dwi_preproc_folder, 'MNI_lin0GenericAffine.mat')
    MNI_lin_FSL_mat = os.path.join(dwi_preproc_folder, 'dwi2MNI_FSL.mat')
    dwi_t1_ANTs_mat = os.path.join(dwi_preproc_folder, 'syn_sdc_wf/ref_2_t1/transform0GenericAffine.mat')
    step2_output_dwi_final_MNI_nii = os.path.join(dwi_preproc_folder, 'Diffusion_iso_MNI.nii.gz')
    step2_output_dwi_final_MNI_bvec = os.path.join(dwi_preproc_folder, 'Diffusion_iso_MNI.bvec')
    step2_output_dwi_final_MNI_bval = os.path.join(dwi_preproc_folder, 'Diffusion_iso_MNI.bval')
    step2_output_dwi_final_MNI_bzero_nii = os.path.join(dwi_preproc_folder, 'Diffusion_iso_b0_MNI.nii.gz')
    step2_output_dwi_final_MNI_mask_nii = os.path.join(dwi_preproc_folder, 'Diffusion_iso_mask_MNI.nii.gz')


    if args.resume and os.path.exists(step2_output_dwi_final_mif): # if -resume, check if output file exist or not
        app_console('Step 2 output found, skip this step')
    else:  # no output file found, begin process
        app_console('run Step 2: dwi round2-preprocessing')

        # Generate an image containing all voxels where the DWI contains valid data
        app_console('Generate an image containing all voxels where the DWI contains valid data')
        dwi_validdata_image = os.path.join(dwi_preproc_folder, 'dwi_validdata_mask.mif')
        command('mrmath ' + step1_output_dwi_image + ' max -axis 3 - |'
                + ' mrthreshold - ' + dwi_validdata_image + ' -abs 0.0 -comparison gt')

        # Determine whether we are working with single-shell or multi-shell data
        bvalues = [int(round(float(value))) for value in mrinfo(step1_output_dwi_image, 'shell_bvalues').strip().split()]
        multishell = (len(bvalues) > 2)
        dwi_image = step1_output_dwi_image
        if t1_dir:
            # use warpped T1 brain mask as DWI brain mask 
            dwi_mask_image = os.path.join(dwi_preproc_folder, 'dwi_mask_init.nii.gz')
            command('antsApplyTransforms -d 3 -i ' + step1_t1_mask_path +
                     ' -r ' + dwi_gibbs_bzero_path + ' -t [' + os.path.join(dwi_preproc_folder, 'syn_sdc_wf/ref_2_t1/transform0GenericAffine.mat') + 
                     ',1]   -n GenericLabel[Linear] -o ' + dwi_mask_image)
            
        else:
            # Initial DWI brain mask
            dwi_mask_image = os.path.join(dwi_preproc_folder, 'dwi_mask_init.mif')
            app_console('Performing intial DWI brain masking')
            command('dwi2mask ' + step1_output_dwi_image + ' ' + dwi_mask_image)
            if args.dwi_mask_path:
                command("mrconvert " + args.dwi_mask_path + " " + dwi_mask_image + " -force")
            else:
                dwi_mask_image = dwi_mask_loop_create(dwi_image, dwi_mask_image, dwi_validdata_image, multishell)

        # Crop images to reduce storage space (but leave some padding on the sides)
        dwi_cropped_image = os.path.join(dwi_preproc_folder, 'dwi_crop.mif')
        dwi_cropped_mask_image = os.path.join(dwi_preproc_folder, 'mask_crop.mif')
        command('mrgrid ' + dwi_image + ' crop ' + dwi_cropped_image
                    + ' -mask ' + dwi_mask_image + ' -uniform -5')
        dwi_image = dwi_cropped_image
        command('mrgrid ' + dwi_mask_image + ' crop ' + dwi_cropped_mask_image +
                ' -mask ' + dwi_mask_image + ' -uniform -5')
        dwi_mask_image = dwi_cropped_mask_image

        # median filter for dwi mask if litter hole occur
        app_warn('perform median filter for dwi mask if litter hole occur (double check)')
        dwi_mask_median_filter = os.path.join(dwi_preproc_folder, 'Diffusion_mask_filtered.mif')
        command('maskfilter ' + dwi_mask_image + ' median -extent 9 ' + dwi_mask_median_filter)
        dwi_mask_image = dwi_mask_median_filter

        # output of dwi_preproc
        command('mrconvert ' + dwi_image + ' '
                + step2_output_dwi_proc_nii
                + ' -export_grad_fsl ' + step2_output_dwi_bvec + ' '
                + step2_output_dwi_bval + ' -strides 1,2,3,4') 
        command('mrconvert ' + dwi_mask_image + ' ' + step2_output_dwi_mask_image + ' -datatype uint8 -strides 1,2,3') 

        command('dwiextract ' + dwi_image + ' -bzero - | '
                    'mrcalc - 0.0 -max - | '
                    'mrmath - mean -axis 3 ' + step2_output_bzero)

        # downsample T1w to 2mm 
        command('mrgrid ' + step1_t1_ss_path  + ' regrid -voxel 2 -interp linear ' + step2_t1_2mm_ss_path)
        command('mrgrid ' + step1_t1_mask_path  + ' regrid -voxel 2 -interp nearest ' + step2_t1_2mm_mask_path)

        # transform dwi to the T1 base space
        command('antsApplyTransforms -d 3 -e 3  -i ' + step2_output_dwi_proc_nii + ' -r ' + 
                step2_t1_2mm_mask_path + ' -t ' + dwi_t1_ANTs_mat + 
                ' -o ' + step2_output_dwi_final_nii + ' -v --float -n LanczosWindowedSinc -f 0')
        command('antsApplyTransforms -d 3 -i ' + step2_output_dwi_mask_image + ' -r ' + 
                    step2_t1_2mm_mask_path + ' -t ' + dwi_t1_ANTs_mat +  
                    ' -o ' + step2_output_dwi_mask_final_nii + ' -v -n GenericLabel[Linear]')
    
        # convert fsl mat to ANTs mat
        command('c3d_affine_tool -ref ' + step1_t1_ss_path + ' -src ' + step2_output_bzero +
                ' -itk ' + dwi_t1_ANTs_mat + ' -ras2fsl -o ' + dwi_t1_FSL_mat)
        # rotate bvec
        rotate_bvecs_func.rotate(step2_output_dwi_bvec, dwi_t1_FSL_mat, step2_output_dwi_final_bvec) 
        # copy bval
        shutil.copyfile(step2_output_dwi_bval, step2_output_dwi_final_bval)
        command('mrconvert ' + step2_output_dwi_final_nii + ' '
                + step2_output_dwi_final_mif
                + ' -fslgrad ' + step2_output_dwi_final_bvec + ' '
                + step2_output_dwi_final_bval)
        command('dwiextract ' + step2_output_dwi_final_mif + ' -bzero - | '
                'mrcalc - 0.0 -max - | '
                'mrmath - mean -axis 3 ' + step2_output_b0_final_nii)
        
        # qc
        bzero_masking_gif = os.path.join(qc_dir, 'bzero_masking.gif')
        command('slices ' + step2_output_b0_final_nii + ' ' + step2_output_dwi_mask_final_nii + ' -o ' + bzero_masking_gif)

        # transform dwi to MNI (linear spatial normalization)
        if t1_dir:
            command('antsRegistrationSyNQuick.sh -d 3 -m ' + step2_output_b0_final_nii +
                     ' -f ' + MNI_T2w_template + ' -x ' + MNI_T2w_template_mask +
                     ' -t r -o ' + os.path.join(dwi_preproc_folder, 'MNI_lin'))
            command('antsApplyTransforms -d 3 -e 3  -i ' + step2_output_dwi_final_nii + ' -r ' + 
                        MNI_T2w_template + ' -t ' + MNI_lin_ANTs_mat +
                        ' -o ' + step2_output_dwi_final_MNI_nii + ' -v --float -n LanczosWindowedSinc -f 0')
            command('antsApplyTransforms -d 3 -i ' + step2_output_dwi_mask_final_nii + ' -r ' + 
                        MNI_T2w_template + ' -t ' + MNI_lin_ANTs_mat +
                        ' -o ' + step2_output_dwi_final_MNI_mask_nii + ' -v -n GenericLabel[Linear]')
            command('antsApplyTransforms -d 3 -i ' + step2_output_b0_final_nii + ' -r ' + 
                        MNI_T2w_template + ' -t ' + MNI_lin_ANTs_mat +
                        ' -o ' + step2_output_dwi_final_MNI_bzero_nii + ' -v --float -n LanczosWindowedSinc -f 0')
            # convert fsl mat to ANTs mat
            command('c3d_affine_tool -ref ' + MNI_T2w_template + ' -src ' + step2_output_b0_final_nii +
                     ' -itk ' + MNI_lin_ANTs_mat + ' -ras2fsl -o ' + MNI_lin_FSL_mat)
            # rotate bvec
            rotate_bvecs_func.rotate(step2_output_dwi_final_bvec, MNI_lin_FSL_mat, step2_output_dwi_final_MNI_bvec)
            # copy bval
            shutil.copyfile(step2_output_dwi_final_bval, step2_output_dwi_final_MNI_bval)

            # qc
            bzero_MNI_gif = os.path.join(qc_dir, 'bzero_in_MNIspace.gif')
            command('slices ' + step2_output_dwi_final_MNI_bzero_nii + ' ' + step2_output_dwi_final_MNI_mask_nii + ' -o ' + bzero_MNI_gif)

    app_console('complete Step 2: dwi round2-preprocessing')
    app_console('-------------------------------------')

    #-----------------------------------------------------------------
    # Final Step: move files to output directory
    #-----------------------------------------------------------------
    app_console('Final Step: move files to output directory')
    if os.path.exists(output_dir):
        app_warn('Found output_dir existing, delete it and create a new one')
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    final_output_dwi_nii = 'dwi.nii.gz'
    final_output_dwi_bvec = 'dwi.bvec'
    final_output_dwi_bval = 'dwi.bval'
    final_output_dwi_mask = 'dwi_mask.nii.gz'
    final_output_dwi_json = 'dwi.json'
    final_output_dwi_b0 = 'dwi_bzero.nii.gz'
    final_output_dwi_MNI_bvec = 'dwi_mni.bvec'
    final_output_dwi_MNI_bval = 'dwi_mni.bval'
    final_output_dwi_MNI_nii = 'dwi_mni.nii.gz'
    final_output_dwi_b0_MNI_nii = 'dwi_mni_bzero.nii.gz'
    final_output_dwi_mask_MNI_nii = 'dwi_mni_mask.nii.gz'
    final_output_dwi2MNI_ANTs_mat = 'dwi_mni_ants.mat'
    final_output_dwi2MNI_FSL_mat = 'dwi_mni_fsl.mat'
    final_output_dwi2T1_ANTs_mat = 'dwi_t1_ants.mat'

    # copy processed dwi files
    shutil.copyfile(step2_output_dwi_final_nii, os.path.join(output_dir, final_output_dwi_nii))
    shutil.copyfile(step2_output_dwi_final_bvec, os.path.join(output_dir, final_output_dwi_bvec))
    shutil.copyfile(step2_output_dwi_final_bval, os.path.join(output_dir, final_output_dwi_bval))
    shutil.copyfile(step2_output_dwi_mask_final_nii, os.path.join(output_dir, final_output_dwi_mask))
    shutil.copyfile(step2_output_b0_final_nii, os.path.join(output_dir, final_output_dwi_b0))

    shutil.copyfile(step2_output_dwi_final_MNI_bvec, os.path.join(output_dir, final_output_dwi_MNI_bvec))
    shutil.copyfile(step2_output_dwi_final_MNI_bval, os.path.join(output_dir, final_output_dwi_MNI_bval))
    shutil.copyfile(step2_output_dwi_final_MNI_nii, os.path.join(output_dir, final_output_dwi_MNI_nii))
    shutil.copyfile(step2_output_dwi_final_MNI_bzero_nii, os.path.join(output_dir, final_output_dwi_b0_MNI_nii))
    shutil.copyfile(step2_output_dwi_final_MNI_mask_nii, os.path.join(output_dir, final_output_dwi_mask_MNI_nii))

    shutil.copyfile(MNI_lin_ANTs_mat, os.path.join(output_dir, final_output_dwi2MNI_ANTs_mat))
    shutil.copyfile(MNI_lin_FSL_mat, os.path.join(output_dir, final_output_dwi2MNI_FSL_mat))

    shutil.copyfile(dwi_json_list[0], os.path.join(output_dir, final_output_dwi_json))
    shutil.copytree(qc_dir, os.path.join(output_dir, 'qc'))

    if os.path.exists(dwi_to_t1_affine):
        shutil.copyfile(dwi_to_t1_affine, os.path.join(output_dir, final_output_dwi2T1_ANTs_mat))

    end = time.time()
    Execution_time = str(round((end - start)/60, 2))
    app_console('dwi preprocessing finished. Execution time: ' + Execution_time)

    if cleanup:
        shutil.rmtree(tempDir)
    tempDir = ''


if __name__ == "__main__":
    global cleanup, resume, tempDir, workingDir
    tempDir = ''

    parser = argparse.ArgumentParser(description="diffusion MRI data preprocessing on participant level",
                                     epilog="Revised from QSIprep and mrtrix3_connectome.")

    parser.add_argument('bids_dir', help='The directory with the input dataset '
                        'formatted according to the BIDS standard.')
    parser.add_argument('output_dir', help='The directory where the output files '
                        'should be stored. If you are running group level analysis '
                        'this folder should be prepopulated with the results of the'
                        'participant level analysis.')
    parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                        'Multiple participant level analyses can be run independently '
                        '(in parallel) using the same output_dir.',
                        choices=['participant', 'group'])
    parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                        'corresponds to sub-<participant_label> from the BIDS spec '
                        '(so it does not include "sub-"). If this parameter is not '
                        'provided all subjects should be analyzed. Multiple '
                        'participants can be specified with a space separated list.',
                        nargs="+")
    parser.add_argument('--session_label', help='The label of the session that should be analyzed. The label '
                        'corresponds to ses-<session_label> from the BIDS spec '
                        '(so it does not include "ses-"). If this parameter is not '
                        'provided, all sessions should be analyzed. Multiple '
                        'sessions can be specified with a space separated list.',
                        nargs="+")
    parser.add_argument("-mode", metavar="type", choices=["fast", "complete"],
                        help="type of dMRI preprocessing, "
                             "fast: dwidenoise (default)."
                             "complete: dwidenoise + gibbs ringing removal + eddy_topup",
                        default='fast')
    parser.add_argument("-resume", action="store_true",
                        help="resume the uncompleted process, for debug only",
                        default=False) 
    parser.add_argument("-cleanup", action="store_true",
                        help="remove temp folder after finish",
                        default=False)
    parser.add_argument('-v', '--version', action='version',
                        version='BIDS-App version {}'.format(__version__))


    args = parser.parse_args()

    workingDir = args.output_dir
    t1prep_dir = os.path.join(args.bids_dir, 'derivatives', 'smri_prep')
    resume = args.resume
    cleanup = args.cleanup
    start = time.time()
    
    # parse bids layout
    layout = bids.layout.BIDSLayout(args.bids_dir, derivatives=False, absolute_paths=True)
    subjects_to_analyze = []

    # only for a subset of subjects
    if args.participant_label:
        subjects_to_analyze = args.participant_label
    # for all subjects
    else:
        subject_dirs = glob.glob(os.path.join(args.bids_dir, "sub-*"))
        subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]
    subjects_to_analyze.sort()

    # only use a subset of sessions
    if args.session_label:
        session_to_analyze = dict(session=args.session_label)
    else:
        session_to_analyze = dict()

   # running participant level
    if args.analysis_level == "participant":
        # find all T1s 
        for subject_label in subjects_to_analyze:
            smri = [f.path for f in layout.get(subject=subject_label,suffix='T1w',extension=["nii.gz", "nii"],**session_to_analyze)]  

        if os.path.normpath(smri[0]).split(os.sep)[-3].split("-")[0] == 'ses':
            sessions = [os.path.normpath(t1).split(os.sep)[-3].split("-")[-1] for t1 in smri]
            sessions.sort()
        else:
            sessions = []

        if sessions:
            for s in range(len(sessions)):  
                session_label = sessions[s]
                smri_analyze = [f.path for f in layout.get(subject=subject_label,session=session_label, suffix='T1w',extension=["nii.gz", "nii"])][0]
                runSubject(args, subject_label, session_label, t1prep_dir)
        else:
            session_label = []
            smri_analyze = smri[0]
            runSubject(args, subject_label, session_label, t1prep_dir)


    # running group level
    elif args.analysis_level == "group":
        if args.participant_label:
            app_error('Cannot use --participant_label option when performing group analysis')
            app_console('Warning: the group analysis is still in development')
            # runGroup(os.path.abspath(args.output_dir))

    app_complete()
    end = time.time()
    running_time = end - start
    print('running time: {:.0f}min {:.0f}sec'.format(running_time//60, running_time % 60))

    