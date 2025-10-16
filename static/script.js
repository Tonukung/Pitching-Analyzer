document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const body = document.body;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        body.style.cursor = 'wait';
        Swal.fire({
            title: 'กำลังวิเคราะห์ข้อมูล...',
            html: 'กรุณารอสักครู่ ระบบกำลังประมวลผลไฟล์เสียงของคุณ',
            allowOutsideClick: false,
            allowEscapeKey: false,
            showConfirmButton: false, 
            width: 600,
            padding: "3em",
            color: "#716add",
            backdrop: `
                rgba(0,0,123,0.4)
                url("https://i.pinimg.com/originals/ed/35/f8/ed35f861be81be2548e514085fb19385.gif")
                center top
                no-repeat
            `
        });
        try {
            await new Promise(resolve => setTimeout(resolve, 1000)); // wait respond API Form [ AI:bell ]
            const response = await fetch('/uploadfile/', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json(); 
            
            Swal.close(); 

            let timerInterval;
            Swal.fire({
            title: "สำเร็จ!",
            html: "ไฟล์ได้รับการอัปโหลดและวิเคราะห์แล้ว",
            icon: 'success',
            timer: 2000,
            didOpen: () => {
                timerInterval = setInterval(() => {
                }, 500);
            },
            willClose: () => {
                clearInterval(timerInterval);
            }
            }).then(() => {
                if (result.redirect) {
                    window.location.href = result.redirect; 
                } else {
                    window.location.href = '/result.html'; // Fallback
                }
            });

        } catch (error) {
            console.error('Upload failed:', error);
            Swal.close(); 
            
            Swal.fire({
                title: 'ข้อผิดพลาด!',
                text: 'การอัปโหลดและวิเคราะห์ล้มเหลว. กรุณาลองใหม่อีกครั้ง.',
                icon: 'error',
                confirmButtonText: 'ตกลง'
            });
            form.reset();

        } finally {
            body.style.cursor = 'default';
        }
    });
});
