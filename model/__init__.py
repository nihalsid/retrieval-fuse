from model.attention import AttentionBlock, PatchedAttentionBlock
from model.refinement import Superresolution08UNetBackbone, SurfaceReconstructionUNetBackbone, Superresolution08FinalDecoder, RetrievalUNetBackbone, Superresolution16UNetBackbone
from model.retrieval import Patch04, Patch08, Patch16, Patch24, Patch32, PCPatch32, PCPatch48, PCPatch64, Patch12, PatchNorm08, PatchNorm32, Patch24V2, Patch04V2


def get_retrieval_networks(model_config):
    fenc_input = fenc_target = None
    if model_config['network_input'] == "2+1":
        fenc_input = Patch04(model_config['nf_input'], model_config['latent_dim'])
    if model_config['network_input'] == "2+1V2":
        fenc_input = Patch04V2(model_config['nf_input'], model_config['latent_dim'])
    elif model_config['network_input'] == "4+2":
        fenc_input = Patch08(model_config['nf_input'], model_config['latent_dim'])
    elif model_config['network_input'] == "4+2N":
        fenc_input = PatchNorm08(model_config['nf_input'], model_config['latent_dim'])
    elif model_config['network_input'] == "16+4":
        fenc_input = Patch24(model_config['nf_input'], model_config['latent_dim'])
    elif model_config['network_input'] == "pc_16+8":
        fenc_input = PCPatch32(model_config['nf_input'], model_config['latent_dim'])
    elif model_config['network_input'] == "pc_32+8":
        fenc_input = PCPatch48(model_config['nf_input'], model_config['latent_dim'])
    elif model_config['network_input'] == "pc_32+16":
        fenc_input = PCPatch64(model_config['nf_input'], model_config['latent_dim'])
    if model_config['network_target'] == "pc_32+16":
        fenc_target = PCPatch64(model_config['nf_target'], model_config['latent_dim'])
    elif model_config['network_target'] == "8+2":
        fenc_target = Patch12(model_config['nf_target'], model_config['latent_dim'])
    elif model_config['network_target'] == "8+4":
        fenc_target = Patch16(model_config['nf_target'], model_config['latent_dim'])
    elif model_config['network_target'] == "16+4":
        fenc_target = Patch24(model_config['nf_target'], model_config['latent_dim'])
    elif model_config['network_target'] == "16+4V2":
        fenc_target = Patch24V2(model_config['nf_target'], model_config['latent_dim'])
    elif model_config['network_target'] == "16+8":
        fenc_target = Patch32(model_config['nf_target'], model_config['latent_dim'])
    elif model_config['network_target'] == "16+8N":
        fenc_target = PatchNorm32(model_config['nf_target'], model_config['latent_dim'])
    return fenc_input, fenc_target


def get_unet_backbone(config):
    if config['task'] == 'superresolution':
        if config['dataset_train']['input_chunk_size'] == 8:
            return Superresolution08UNetBackbone(config['nf'], num_levels=config['unet_num_level'], layer_order=config['layer_order'])
        elif config['dataset_train']['input_chunk_size'] == 16:
            return Superresolution16UNetBackbone(config['nf'], num_levels=config['unet_num_level'], layer_order=config['layer_order'])
    if config['task'] == 'surface_reconstruction':
        return SurfaceReconstructionUNetBackbone(config['nf'], num_levels=config['unet_num_level'], layer_order=config['layer_order'])


def get_decoder(config):
    return Superresolution08FinalDecoder(config['nf'], layer_order=config['layer_order'])


def get_retrieval_backbone(config):
    return RetrievalUNetBackbone(nf=config['nf'], f_maps=config['retrieval_fmaps'], num_levels=config['retrieval_num_level'], layer_order=config['layer_order'])


def get_attention_block(config):
    attention_block = AttentionBlock(config['nf'], config['attn_patch_extent'] // 2, config['K'], config['attn_normalize'], config['attn_use_switching'], config['attn_retrieval_mode'], config['attn_no_output_mapping'], config['attn_blend'])
    return PatchedAttentionBlock(config['nf'], config['attn_num_patch'], config['attn_patch_extent'] // 2, config['K'], attention_block)
