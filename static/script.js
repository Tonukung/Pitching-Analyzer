document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("audio-upload");
    const file = fileInput.files[0];
    if (!file) {
      Swal.fire("กรุณาเลือกไฟล์ก่อน", "", "warning");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    Swal.fire({
      title: "กำลังประมวลผล...",
      text: "กรุณารอสักครู่ ระบบกำลังวิเคราะห์ไฟล์เสียงของคุณ",
      allowOutsideClick: false,
      didOpen: () => {
        Swal.showLoading();
      }
    });

    try {
      const response = await fetch("/uploadfile/", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      // ตรวจสอบว่ามี error หรือไม่
      if (!response.ok || data.redirect === null || data.redirect === undefined) {
        Swal.fire({
          icon: "error",
          title: "การวิเคราะห์ล้มเหลว",
          text: data.api_result?.error
            ? data.api_result.error
            : "ไม่สามารถเชื่อมต่อกับ API วิเคราะห์ได้ กรุณาลองใหม่อีกครั้ง"
        });
        return;
      }

      Swal.fire({
        icon: "success",
        title: "สำเร็จ!",
        text: data.message
      }).then(() => {
        window.location.href = data.redirect;
      });

    } catch (error) {
      Swal.fire({
        icon: "error",
        title: "เกิดข้อผิดพลาด",
        text: "ไม่สามารถติดต่อเซิร์ฟเวอร์ได้ กรุณาลองใหม่อีกครั้ง"
      });
      console.error("Error during upload:", error);
    }
  });
});
