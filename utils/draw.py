import matplotlib.pyplot as plt

def show_attention(attention):
    """
    画出注意力权重图

    Parameters
    ----------
    @param attention: torch.Tensor
        attention weights, shape (m, n)
    """

    fig = plt.figure(figsize=(5, 5))

    pcm = plt.imshow(
        attention.detach().numpy(), 
        cmap='Reds'
    )

    plt.xlabel('Key points')
    plt.ylabel('Query points')
    plt.title('Attention weights')
    fig.colorbar(pcm, shrink=0.7)
    plt.show()