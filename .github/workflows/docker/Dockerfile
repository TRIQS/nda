FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
    ca-certificates \
    build-essential \
    ccache \
    cmake \
    g++ \
    git \
    libhdf5-dev \
    libopenblas-dev \
    libopenmpi-dev \
    openmpi-bin \
    openssh-server \
    python3-dev \
    python3-mako \
    python3-mpi4py \
    python3-numpy \
    python3-scipy \
 && rm -rf /var/cache/apt /var/lib/apt/lists
RUN useradd -m runner \
 && passwd -d runner \
 && mkdir -pm0755 /run/sshd \
 && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config.d/99-insecure.conf \
 && echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config.d/99-insecure.conf \
 && echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config.d/99-insecure.conf \
 && echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config.d/99-insecure.conf \
 && echo "LogLevel ERROR" >> /etc/ssh/ssh_config.d/99-insecure.conf
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV OMPI_MCA_osc=sm,pt2pt
ENV OMPI_MCA_rmaps_base_oversubscribe=yes
CMD ["/usr/sbin/sshd", "-D"]
