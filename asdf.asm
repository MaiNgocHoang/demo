org 100h

; In thong bao yeu cau nhap 5 so
mov ah, 09h
lea dx, nhapso
int 21h

; Nhap 5 so
mov cx, 5
Nhap_So:
    mov ah, 01h
    int 21h
    sub al, '0'
    push ax
    loop Nhap_So

; In day so nguoc lai
mov ah, 09h
lea dx, xuat5so
int 21h

pop dx
In_So:
    add dl, '0'
    mov ah, 02h
    int 21h
    pop dx
    loop In_So

; Dung man hinh
mov ah, 0h
int 16h

ret

nhapso db "Nhap 5 so theo day: $"
xuat5so db 0ah, 0dh, "Day so theo chieu nguoc lai: $"
