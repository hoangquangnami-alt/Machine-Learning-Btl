"""Nhãn hiển thị cho mã phân loại UCI; không thay đổi dữ liệu đầu vào mô hình.
Nguồn: https://archive.ics.uci.edu/dataset/697
"""
YES_NO = {0: 'Không', 1: 'Có'}
QUALIFICATION = {
 1:'Trung học phổ thông hoặc tương đương', 2:'Đại học - bằng cử nhân', 3:'Đại học - bằng tốt nghiệp',
 4:'Thạc sĩ', 5:'Tiến sĩ', 6:'Đã học đại học', 9:'Chưa hoàn thành lớp 12',
 10:'Chưa hoàn thành lớp 11', 11:'Lớp 7 (chương trình cũ)', 12:'Khác - lớp 11',
 13:'Năm thứ hai trung học bổ túc', 14:'Lớp 10', 15:'Chưa hoàn thành lớp 10',
 18:'Khóa thương mại tổng hợp', 19:'Giáo dục cơ bản chu kỳ 3 hoặc tương đương',
 20:'Khóa trung học bổ túc', 22:'Khóa kỹ thuật - nghề', 25:'Chưa hoàn thành trung học bổ túc',
 26:'Lớp 7', 27:'Chu kỳ 2 trung học phổ thông', 29:'Chưa hoàn thành lớp 9', 30:'Lớp 8',
 31:'Khóa quản trị và thương mại', 33:'Khóa bổ túc kế toán và quản trị', 34:'Không rõ',
 35:'Không biết đọc hoặc viết', 36:'Biết đọc, chưa hoàn thành lớp 4',
 37:'Giáo dục cơ bản chu kỳ 1 hoặc tương đương', 38:'Giáo dục cơ bản chu kỳ 2 hoặc tương đương',
 39:'Khóa chuyên môn công nghệ', 40:'Đại học - chu kỳ 1', 41:'Khóa học chuyên ngành cao đẳng',
 42:'Khóa kỹ thuật chuyên nghiệp bậc cao', 43:'Thạc sĩ - chu kỳ 2', 44:'Tiến sĩ - chu kỳ 3'}
OCCUPATION = {0:'Học sinh/sinh viên', 1:'Lãnh đạo, quản lý', 2:'Chuyên gia khoa học và trí tuệ',
 3:'Kỹ thuật viên và nghề trình độ trung cấp', 4:'Nhân viên hành chính', 5:'Dịch vụ cá nhân, an ninh hoặc bán hàng',
 6:'Lao động có tay nghề nông, lâm, ngư nghiệp', 7:'Lao động có tay nghề công nghiệp, xây dựng, thủ công',
 8:'Vận hành máy móc, lắp ráp', 9:'Lao động phổ thông', 10:'Lực lượng vũ trang', 90:'Tình trạng khác', 99:'Không có thông tin',
 122:'Chuyên gia y tế',123:'Giáo viên',125:'Chuyên gia công nghệ thông tin',
 131:'Kỹ thuật viên khoa học và kỹ thuật',132:'Kỹ thuật viên y tế',134:'Kỹ thuật viên pháp lý, xã hội, văn hóa',
 141:'Nhân viên văn phòng, thư ký, xử lý dữ liệu',143:'Nhân viên dữ liệu, kế toán, thống kê, tài chính',
 144:'Nhân viên hỗ trợ hành chính khác',151:'Nhân viên dịch vụ cá nhân',152:'Nhân viên bán hàng',
 153:'Nhân viên chăm sóc cá nhân',171:'Lao động xây dựng có tay nghề',
 173:'Lao động in ấn, chế tạo dụng cụ chính xác, thủ công',175:'Lao động chế biến thực phẩm, gỗ, may mặc',
 191:'Nhân viên vệ sinh',192:'Lao động phổ thông nông, lâm, ngư nghiệp',
 193:'Lao động phổ thông khai thác, xây dựng, sản xuất, vận tải',194:'Phụ bếp'}
LABELS = {
 'Marital status':{1:'Độc thân',2:'Đã kết hôn',3:'Góa',4:'Đã ly hôn',5:'Chung sống thực tế',6:'Ly thân hợp pháp'},
 'Application mode':{1:'Đợt 1 - tuyển sinh chung',2:'Quy định 612/93',5:'Đợt 1 - diện đặc biệt Azores',7:'Có bằng đại học khác',10:'Quy định 854-B/99',15:'Sinh viên quốc tế - cử nhân',16:'Đợt 1 - diện đặc biệt Madeira',17:'Đợt 2 - tuyển sinh chung',18:'Đợt 3 - tuyển sinh chung',26:'Quy định 533-A/99 - chương trình khác',27:'Quy định 533-A/99 - trường khác',39:'Trên 23 tuổi',42:'Chuyển trường',43:'Chuyển ngành',44:'Có bằng chuyên môn công nghệ',51:'Chuyển trường hoặc ngành',53:'Có bằng chương trình ngắn hạn',57:'Chuyển trường hoặc ngành quốc tế'},
 'Course':{33:'Công nghệ nhiên liệu sinh học',171:'Thiết kế hoạt hình và đa phương tiện',8014:'Công tác xã hội - buổi tối',9003:'Nông học',9070:'Thiết kế truyền thông',9085:'Điều dưỡng thú y',9119:'Kỹ thuật tin học',9130:'Chăn nuôi ngựa',9147:'Quản lý',9238:'Công tác xã hội',9254:'Du lịch',9500:'Điều dưỡng',9556:'Vệ sinh răng miệng',9670:'Quản lý quảng cáo và tiếp thị',9773:'Báo chí và truyền thông',9853:'Giáo dục cơ bản',9991:'Quản lý - buổi tối'},
 'Daytime/evening attendance':{0:'Buổi tối',1:'Ban ngày'},
 'Previous qualification':QUALIFICATION, "Mother's qualification":QUALIFICATION, "Father's qualification":QUALIFICATION,
 "Mother's occupation":OCCUPATION, "Father's occupation":OCCUPATION,
 'Nacionality':{1:'Bồ Đào Nha',2:'Đức',6:'Tây Ban Nha',11:'Ý',13:'Hà Lan',14:'Anh',17:'Litva',21:'Angola',22:'Cape Verde',24:'Guinea',25:'Mozambique',26:'São Tomé và Príncipe',32:'Thổ Nhĩ Kỳ',41:'Brazil',62:'Romania',100:'Moldova',101:'Mexico',103:'Ukraine',105:'Nga',108:'Cuba',109:'Colombia'},
 'Displaced':YES_NO,'Educational special needs':YES_NO,'Gender':{0:'Nữ',1:'Nam'},'International':YES_NO,
}
def label_for(column, code):
    return LABELS.get(column, {}).get(code, f'Mã {code} — xem bảng mã UCI')
