# Google Gemini TTS Python

Chuyển đổi văn bản tiếng Việt hoặc đa ngôn ngữ thành giọng nói tự động bằng API Gemini 2.5 Flash (Google AI), xuất ra file `.wav` trên máy tính.

## Tính năng

- Đọc nội dung từ file `input.txt` và chuyển thành audio giọng tự nhiên (hỗ trợ tiếng Việt).
- Lưu output thành file `.wav`, đặt tên theo thời gian.
- Hiển thị tiến trình tạo audio bằng progress bar (tqdm).
- Đảm bảo bảo mật: `.env` để lưu API key, không đưa vào repo.

## Cài đặt

1. **Clone repo:**
    ```sh
    git clone https://github.com/thanh-abaii/googletts.git
    cd googletts
    ```

2. **Tạo và kích hoạt môi trường ảo Python (khuyên dùng):**
    ```sh
    python -m venv .venv
    .venv\Scripts\activate    # Windows
    # source .venv/bin/activate   # Linux/Mac
    ```

3. **Cài các thư viện cần thiết:**
    ```sh
    pip install -r requirements.txt
    ```
    Hoặc cài từng thư viện:
    ```sh
    pip install google-genai python-dotenv tqdm
    ```

4. **Tạo file `.env` chứa API key:**
    ```
    GEMINI_API_KEY=your-google-gemini-api-key
    ```

5. **Đảm bảo đã có file `input.txt` chứa nội dung muốn chuyển thành giọng nói.**

## Sử dụng

```sh python main.py```

    Kết quả sẽ được lưu thành file .wav trong thư mục dự án, tên dạng output_puck_YYYYMMDD_HHMMSS.wav.
    Tiến trình chuyển đổi sẽ hiển thị progress bar theo từng chunk dữ liệu nhận được từ API.

## Lưu ý bảo mật
- KHÔNG đưa file .env, file âm thanh .wav hoặc thư mục .venv/ lên Github.
- Đã cấu hình .gitignore chuẩn cho Python project.

## Giới hạn
- Google Gemini TTS hiện chỉ hỗ trợ sinh tối đa ~5 phút audio mỗi lần gọi API. Văn bản dài hơn cần chia nhỏ và xử lý nhiều lần.

## Đóng góp & bản quyền
- Repo do thanh-abaii phát triển và chia sẻ vì mục đích học thuật, nghiên cứu, không thương mại.
- Mọi ý kiến đóng góp hoặc vấn đề vui lòng mở issue hoặc liên hệ qua Github.