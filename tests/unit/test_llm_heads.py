from decoder_transformer.llm_heads import SingleHead, FeedForward, BlockSH, BlockMH, MultiHead, Block, MultiHeadwithEncoder
import pytest
import torch



class TestDecoderHeads:

    def test_single_head(self):

        single_head = SingleHead(512, 512, 256)
        input = torch.randn(size = (32, 256, 512))

        output = single_head(input)

        assert output.shape == (32, 256, 512)

    def test_ffn(self):

        ffn = FeedForward(512)
        input = torch.randn(size = (32, 256, 512))
        output = ffn(input)

        assert output.shape == (32, 256, 512)

    def test_multihead(self):

        mhead = MultiHead(4, 128, 256)
        input = torch.randn(size = (32, 256, 512))

        output = mhead(input)

        assert output.shape == (32, 256, 512)

    def test_block(self):

        block = Block(512, 4, 256)
        input = torch.randn(size = (32, 256, 512))

        output = block(input)

        assert output.shape == (32, 256, 512)

        block = BlockSH(512, 4, 256)
        input = torch.randn(size = (32, 256, 512))

        output = block(input)

        assert output.shape == (32, 256, 512)

        block = BlockMH(512, 4, 256)
        input = torch.randn(size = (32, 256, 512))

        output = block(input)

        assert output.shape == (32, 256, 512)



    def test_mhencoder(self):

        block = MultiHeadwithEncoder(4, 128, 256)
        input = torch.randn(size = (32, 256, 512))
        output = block(input)
        assert output.shape == (32, 256, 512)
