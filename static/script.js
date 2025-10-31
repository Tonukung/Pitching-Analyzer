document.addEventListener("DOMContentLoaded", () => {
  const pollInterval = 3000; // Check every 3 seconds

  // Function to poll for analysis status
  function pollStatus(filename) {
    const intervalId = setInterval(async () => {
      try {
        const response = await fetch(`/check_status?filename=${filename}`);
        const data = await response.json();

        if (data.status === 'complete') {
          clearInterval(intervalId);
          window.location.href = `/result.html?filename=${filename}`;
        }
      } catch (error) {
        console.error("Polling error:", error);
      }
    }, pollInterval);
  }

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
      text: "กรุณารอสักครู่ ระบบกำลังอัพโหลดไฟล์ของคุณ",
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

      if (!response.ok) {
        Swal.fire({
          icon: "error",
          title: "การวิเคราะห์ล้มเหลว",
          text: data.detail || "ไม่สามารถเชื่อมต่อกับ API วิเคราะห์ได้ กรุณาลองใหม่อีกครั้ง"
        });
        return;
      }

      Swal.fire({ 
        title: "กำลังประมวลผล...",
        text: "อดทนและใจเย็น ๆ ก่อน AI กำลังวิเคราะห์ไฟล์ของคุณ",
        allowOutsideClick: false,
        showConfirmButton: false,
        didOpen: () => {
          Swal.showLoading();
        },
        backdrop: `
        rgba(0,0,123,0.4)
        url("https://raw.githubusercontent.com/gist/brudnak/aba00c9a1c92d226f68e8ad8ba1e0a40/raw/e1e4a92f6072d15014f19aa8903d24a1ac0c41a4/nyan-cat.gif")
        left top
        no-repeat
        `
      });
      pollStatus(data.filename);

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
