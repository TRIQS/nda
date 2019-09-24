	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 14	sdk_version 10, 14
	.intel_syntax noprefix
	.globl	__Z3it2RN3nda5arrayIdLi1EEES2_S2_ ## -- Begin function _Z3it2RN3nda5arrayIdLi1EEES2_S2_
	.p2align	4, 0x90
__Z3it2RN3nda5arrayIdLi1EEES2_S2_:      ## @_Z3it2RN3nda5arrayIdLi1EEES2_S2_
	.cfi_startproc
## %bb.0:
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset rbp, -16
	mov	rbp, rsp
	.cfi_def_cfa_register rbp
	mov	r8, qword ptr [rdi]
	test	r8, r8
	jle	LBB0_17
## %bb.1:
	mov	rcx, qword ptr [rdi + 24]
	mov	rsi, qword ptr [rsi + 24]
	mov	rdx, qword ptr [rdx + 24]
	cmp	r8, 15
	ja	LBB0_8
## %bb.2:
	xor	edi, edi
LBB0_3:
	mov	r9, rdi
	not	r9
	add	r9, r8
	mov	rax, r8
	and	rax, 7
	je	LBB0_6
## %bb.4:
	neg	rax
	.p2align	4, 0x90
LBB0_5:                                 ## =>This Inner Loop Header: Depth=1
	vmovsd	xmm0, qword ptr [rsi + 8*rdi] ## xmm0 = mem[0],zero
	vaddsd	xmm0, xmm0, qword ptr [rdx + 8*rdi]
	vmovsd	qword ptr [rcx + 8*rdi], xmm0
	add	rdi, 1
	inc	rax
	jne	LBB0_5
LBB0_6:
	cmp	r9, 7
	jb	LBB0_17
	.p2align	4, 0x90
LBB0_7:                                 ## =>This Inner Loop Header: Depth=1
	vmovsd	xmm0, qword ptr [rsi + 8*rdi] ## xmm0 = mem[0],zero
	vaddsd	xmm0, xmm0, qword ptr [rdx + 8*rdi]
	vmovsd	qword ptr [rcx + 8*rdi], xmm0
	vmovsd	xmm0, qword ptr [rsi + 8*rdi + 8] ## xmm0 = mem[0],zero
	vaddsd	xmm0, xmm0, qword ptr [rdx + 8*rdi + 8]
	vmovsd	qword ptr [rcx + 8*rdi + 8], xmm0
	vmovsd	xmm0, qword ptr [rsi + 8*rdi + 16] ## xmm0 = mem[0],zero
	vaddsd	xmm0, xmm0, qword ptr [rdx + 8*rdi + 16]
	vmovsd	qword ptr [rcx + 8*rdi + 16], xmm0
	vmovsd	xmm0, qword ptr [rsi + 8*rdi + 24] ## xmm0 = mem[0],zero
	vaddsd	xmm0, xmm0, qword ptr [rdx + 8*rdi + 24]
	vmovsd	qword ptr [rcx + 8*rdi + 24], xmm0
	vmovsd	xmm0, qword ptr [rsi + 8*rdi + 32] ## xmm0 = mem[0],zero
	vaddsd	xmm0, xmm0, qword ptr [rdx + 8*rdi + 32]
	vmovsd	qword ptr [rcx + 8*rdi + 32], xmm0
	vmovsd	xmm0, qword ptr [rsi + 8*rdi + 40] ## xmm0 = mem[0],zero
	vaddsd	xmm0, xmm0, qword ptr [rdx + 8*rdi + 40]
	vmovsd	qword ptr [rcx + 8*rdi + 40], xmm0
	vmovsd	xmm0, qword ptr [rsi + 8*rdi + 48] ## xmm0 = mem[0],zero
	vaddsd	xmm0, xmm0, qword ptr [rdx + 8*rdi + 48]
	vmovsd	qword ptr [rcx + 8*rdi + 48], xmm0
	vmovsd	xmm0, qword ptr [rsi + 8*rdi + 56] ## xmm0 = mem[0],zero
	vaddsd	xmm0, xmm0, qword ptr [rdx + 8*rdi + 56]
	vmovsd	qword ptr [rcx + 8*rdi + 56], xmm0
	add	rdi, 8
	cmp	r8, rdi
	jne	LBB0_7
	jmp	LBB0_17
LBB0_8:
	lea	rdi, [rcx + 8*r8]
	lea	rax, [rsi + 8*r8]
	lea	r9, [rdx + 8*r8]
	cmp	rcx, rax
	setb	r10b
	cmp	rsi, rdi
	setb	r11b
	cmp	rcx, r9
	setb	al
	cmp	rdx, rdi
	setb	r9b
	xor	edi, edi
	test	r10b, r11b
	jne	LBB0_3
## %bb.9:
	and	al, r9b
	jne	LBB0_3
## %bb.10:
	mov	rdi, r8
	and	rdi, -16
	lea	r10, [rdi - 16]
	mov	r11, r10
	shr	r11, 4
	add	r11, 1
	mov	r9d, r11d
	and	r9d, 1
	test	r10, r10
	je	LBB0_11
## %bb.12:
	mov	eax, 1
	sub	rax, r11
	lea	r10, [r9 + rax]
	add	r10, -1
	xor	eax, eax
	.p2align	4, 0x90
LBB0_13:                                ## =>This Inner Loop Header: Depth=1
	vmovupd	ymm0, ymmword ptr [rsi + 8*rax]
	vmovupd	ymm1, ymmword ptr [rsi + 8*rax + 32]
	vmovupd	ymm2, ymmword ptr [rsi + 8*rax + 64]
	vmovupd	ymm3, ymmword ptr [rsi + 8*rax + 96]
	vaddpd	ymm0, ymm0, ymmword ptr [rdx + 8*rax]
	vaddpd	ymm1, ymm1, ymmword ptr [rdx + 8*rax + 32]
	vaddpd	ymm2, ymm2, ymmword ptr [rdx + 8*rax + 64]
	vaddpd	ymm3, ymm3, ymmword ptr [rdx + 8*rax + 96]
	vmovupd	ymmword ptr [rcx + 8*rax], ymm0
	vmovupd	ymmword ptr [rcx + 8*rax + 32], ymm1
	vmovupd	ymmword ptr [rcx + 8*rax + 64], ymm2
	vmovupd	ymmword ptr [rcx + 8*rax + 96], ymm3
	vmovupd	ymm0, ymmword ptr [rsi + 8*rax + 128]
	vmovupd	ymm1, ymmword ptr [rsi + 8*rax + 160]
	vmovupd	ymm2, ymmword ptr [rsi + 8*rax + 192]
	vmovupd	ymm3, ymmword ptr [rsi + 8*rax + 224]
	vaddpd	ymm0, ymm0, ymmword ptr [rdx + 8*rax + 128]
	vaddpd	ymm1, ymm1, ymmword ptr [rdx + 8*rax + 160]
	vaddpd	ymm2, ymm2, ymmword ptr [rdx + 8*rax + 192]
	vaddpd	ymm3, ymm3, ymmword ptr [rdx + 8*rax + 224]
	vmovupd	ymmword ptr [rcx + 8*rax + 128], ymm0
	vmovupd	ymmword ptr [rcx + 8*rax + 160], ymm1
	vmovupd	ymmword ptr [rcx + 8*rax + 192], ymm2
	vmovupd	ymmword ptr [rcx + 8*rax + 224], ymm3
	add	rax, 32
	add	r10, 2
	jne	LBB0_13
## %bb.14:
	test	r9, r9
	je	LBB0_16
LBB0_15:
	vmovupd	ymm0, ymmword ptr [rsi + 8*rax]
	vmovupd	ymm1, ymmword ptr [rsi + 8*rax + 32]
	vmovupd	ymm2, ymmword ptr [rsi + 8*rax + 64]
	vmovupd	ymm3, ymmword ptr [rsi + 8*rax + 96]
	vaddpd	ymm0, ymm0, ymmword ptr [rdx + 8*rax]
	vaddpd	ymm1, ymm1, ymmword ptr [rdx + 8*rax + 32]
	vaddpd	ymm2, ymm2, ymmword ptr [rdx + 8*rax + 64]
	vaddpd	ymm3, ymm3, ymmword ptr [rdx + 8*rax + 96]
	vmovupd	ymmword ptr [rcx + 8*rax], ymm0
	vmovupd	ymmword ptr [rcx + 8*rax + 32], ymm1
	vmovupd	ymmword ptr [rcx + 8*rax + 64], ymm2
	vmovupd	ymmword ptr [rcx + 8*rax + 96], ymm3
LBB0_16:
	cmp	r8, rdi
	jne	LBB0_3
LBB0_17:
	pop	rbp
	vzeroupper
	ret
LBB0_11:
	xor	eax, eax
	test	r9, r9
	jne	LBB0_15
	jmp	LBB0_16
	.cfi_endproc
                                        ## -- End function

.subsections_via_symbols
