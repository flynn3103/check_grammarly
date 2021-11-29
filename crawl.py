import scrapy
from scrapy.crawler import CrawlerProcess
from crochet import setup
import scrapy.crawler as crawler
from multiprocessing import Process, Queue
from twisted.internet import reactor
from utils import _read_file

class VNExpressSpider(scrapy.Spider):
  name = 'vnexpress_crawler'
  allowed_domains = ['vnexpress.net']
  start_urls = ['https://vnexpress.net/cong-tac-nhan-su-dai-hoi-xiii-qua-cac-hoi-nghi-trung-uong-4206324.html', 
                'https://vnexpress.net/cuu-chanh-an-tand-ha-tinh-bi-de-nghi-ky-luat-4206796.html', 
                'https://vnexpress.net/dac-quyen-cua-pho-tong-thong-my-4206671.html', 
                'https://vnexpress.net/pho-wall-lao-doc-vi-noi-so-covid-19-tro-lai-4206370.html', 
                'https://vnexpress.net/cai-thien-lan-da-voi-sua-tam-trang-yukina-4205835.html', 
                'https://vnexpress.net/liverpool-tottenham-tranh-ngoi-dau-ngoai-hang-anh-4206590.html', 
                'https://vnexpress.net/tai-xe-tong-nu-tiep-vien-hang-khong-bi-de-nghi-6-7-nam-tu-4206874.html', 
                'https://vnexpress.net/nguoi-dan-am-uc-vi-du-khach-duoc-di-lai-khap-noi-4206536.html',
                'https://vnexpress.net/lat-nguoc-the-co-nho-chuyen-doi-so-4206387.html', 
                'https://vnexpress.net/co-gai-lac-quan-manh-me-4206416.html', 
                'https://vnexpress.net/canh-sat-giao-thong-tap-trung-xu-ly-vi-pham-dip-cuoi-nam-4206636.html', 
                'https://vnexpress.net/khoa-hoc-lai-may-bay-tai-royhle-flight-4171655.html', 
                'https://vnexpress.net/ba-clinton-keu-goi-bo-he-thong-dai-cu-tri-4206787.html']

  def parse(self, response):
    for next_page in response.css("li > a::attr(href)").re(r'https.*'):
      yield response.follow(next_page, self.parse, )
    for para in [ response.css('section').css('h1::text').get() + ' ', ] + response.css('section').css('p::text').getall():
      with open('/content/drive/MyDrive/nlp projects/Text correction/rawtext3.txt', 'a', encoding = 'utf-8') as f:
        f.write(para)

class ThanhnienSpider(scrapy.Spider):
    #response.css("a::attr(href)").re(r'https.*')
    name = 'baothanhnien_crawler'
    allowed_domains = ['thanhnien.vn']
    start_urls = ['https://thanhnien.vn/doi-song/tet-co-truyen-va-nhung-gia-tri-con-mai-voi-thoi-gian-1322563.html', 
                'https://thanhnien.vn/doi-song/300-trieu-dong-huong-ve-dong-bao-bi-lu-lut-o-mien-trung-1290290.html',
                'https://thanhnien.vn/tai-chinh-kinh-doanh/kinh-te-xanh/ho-bien-mui-hoi-phan-bo-bac-si-thu-y-thanh-ti-phu-1320595.html',
                'https://thanhnien.vn/tai-chinh-kinh-doanh/xe-da-chay-duoc-40-kmh-tren-tuyen-cao-toc-trung-luong-my-thuan-1322564.html',
                'https://thanhnien.vn/tai-chinh-kinh-doanh/nhieu-doanh-nghiep-fdi-doanh-thu-tang-lo-tang-theo-1322260.html', 
                'https://thanhnien.vn/giao-duc/tin-tuc-giao-duc-dac-biet-tren-bao-in-ngay-mai-29122020-1322638.html',
                'https://thanhnien.vn/du-lich/nong-tren-mang-xa-hoi-panorama-ma-pi-leng-sau-cai-tao-lai-hoanh-trang-hon-1320407.html',
                'https://thanhnien.vn/van-hoa/chu-tich-ho-chi-minh-su-dung-nguoi-tai-lam-tuong-1322482.html', 
                'https://thanhnien.vn/suc-khoe/lam-dep/nhung-thong-tin-can-biet-ve-phuong-phap-cat-mi-mat-bi-sup-1322442.html']

    def parse(self, response):
        for next_page in response.css("a::attr(href)").re(r'https.*'):
            yield response.follow(next_page, self.parse, )
        title = [x for x in response.css('h1::text').getall() if x is not None and x != 'None' and len(x.split()) > 5]
        h2 = [x for x in response.css('h2::text').getall() if x is not None and x != 'None' and len(x.split()) > 5]
        body =  [x for x in response.css("div::text").getall() if x is not None and x != 'None' and len(x.split()) > 5 ]
        with open('/content/drive/MyDrive/nlp projects/Text correction/baothanhnien/rawtext.txt', 'a', encoding = 'utf-8') as f:
            f.writelines([para+ '\n' for para in title + h2 + body])

class VietnamnetSpider(scrapy.Spider):
    #response.css("a::attr(href)").re(r'https.*')
    name = 'vietnamnet_crawler'
    allowed_domains = ['vietnamnet.vn']
    start_urls = ['https://vietnamnet.vn/vn/kinh-doanh/dau-tu/khong-doi-moi-sang-tao-se-mac-ket-trong-ho-nang-suat-thap-704159.html', 
                  'https://vietnamnet.vn/vn/thoi-su/tp-hcm-bon-lan-dieu-chinh-nhiem-vu-cua-cac-pho-chu-tich-704110.html', 
                  'https://vietnamnet.vn/vn/thoi-su/bat-giam-truong-khoa-cung-2-dieu-duong-benh-vien-da-khoa-tu-quang-nam-704043.html', 
                  'https://vietnamnet.vn/vn/thoi-su/quoc-hoi/phat-bieu-cua-chu-tich-qh-tai-cuoc-gap-mat-dbqh-qua-cac-thoi-ky-703465.html', 
                  'https://vietnamnet.vn/vn/thoi-su/an-toan-giao-thong/o-to-lao-vao-cho-o-hai-phong-7-nguoi-bi-thuong-704215.html', 
                  'https://vietnamnet.vn/vn/thoi-su/moi-truong/chi-41-ty-dong-tp-vinh-om-84-nghin-tan-rac-thai-chua-xu-ly-703736.html', 
                  'https://vietnamnet.vn/vn/thoi-su/tin-anh/dien-tap-chong-khung-bo-bao-ve-dai-hoi-dang-704005.html', 
                  'https://vietnamnet.vn/vn/thoi-su/bhxh-bhyt/bhxh-viet-nam-trien-khai-nhiem-vu-nam-2021-702207.html', 
                  'https://vietnamnet.vn/vn/thoi-su/chinh-tri/bo-quoc-phong-phat-dong-toan-dan-to-giac-nguoi-nhap-canh-trai-phep-703960.html', 
                  'https://vietnamnet.vn/vn/kinh-doanh/tai-chinh/gia-bitcoin-tang-manh-704100.html', 
                  'https://vietnamnet.vn/vn/kinh-doanh/dau-tu/bat-chap-kho-khan-doanh-nghiep-dich-vu-va-cong-nghiep-van-tiep-tuc-lon-manh-704045.html', 
                  'https://vietnamnet.vn/vn/kinh-doanh/thi-truong/dao-co-thu-gia-tram-trieu-dong-704152.html', 
                  'https://vietnamnet.vn/vn/kinh-doanh/doanh-nhan/co-phieu-tesla-dua-elon-musk-thanh-ty-phu-giau-nhat-hanh-tinh-704151.html', 
                  'https://vietnamnet.vn/vn/kinh-doanh/tu-van-tai-chinh/cach-tinh-thue-thu-nhap-ca-nhan-tu-tien-luong-tien-cong-704148.html', 
                  'https://vietnamnet.vn/vn/kinh-doanh/amaccao/cong-be-tong-amaccao-ran-chac-nhu-da-ben-vinh-cuu-703805.html', 
                  'https://vietnamnet.vn/vn/giai-tri/the-gioi-sao/madonna-la-dien-vien-te-nhat-thap-ky-704102.html', 
                  'https://vietnamnet.vn/vn/giai-tri/thoi-trang/nu-nguoi-mau-trung-quoc-qua-doi-tren-may-bay-704113.html', 
                  'https://vietnamnet.vn/vn/giai-tri/nhac/cuong-seven-ra-mat-mv-anh-da-noi-em-roi-704131.html', 
                  'https://vietnamnet.vn/vn/giai-tri/phim/bat-mi-phim-ve-phi-cong-co-thanh-son-binh-an-dong-phat-tren-vtv-704107.html',
                  'https://vietnamnet.vn/vn/giai-tri/truyen-hinh/duong-trieu-vu-toi-duoc-sinh-ra-o-chuong-heo-704118.html',
                  'https://vietnamnet.vn/vn/giai-tri/sach/hieu-ve-nuoc-va-thien-nhien-de-cai-thien-chat-luong-song-704123.html', 
                  'https://vietnamnet.vn/vn/giai-tri/di-san-my-thuat-san-khau/nsnd-quoc-anh-tra-my-thanh-cap-vo-chong-buoc-vao-cuoc-dua-du-tet-704035.html',
                  'https://vietnamnet.vn/vn/giai-tri/di-san-my-thuat-san-khau/nsnd-quoc-anh-tra-my-thanh-cap-vo-chong-buoc-vao-cuoc-dua-du-tet-704035.html',
                  'https://vietnamnet.vn/vn/the-gioi/huyen-thoai-thoi-trang-pierre-cardin-qua-doi-701480.html', 
                  'https://vietnamnet.vn/vn/the-gioi/ho-so/ly-do-dich-covid-19-khong-tro-thanh-tham-hoa-o-chau-phi-703078.html',
                  'https://vietnamnet.vn/vn/the-gioi/the-gioi-do-day/hung-bao-chi-trich-con-gai-ong-trump-voi-xoa-tweet-ve-nguoi-bieu-tinh-703616.html',
                  'https://vietnamnet.vn/vn/the-gioi/viet-nam-va-the-gioi/chien-thang-cua-long-yeu-nuoc-tinh-ban-vo-tu-trong-sang-viet-nam-campuchia-703780.html',
                  'https://vietnamnet.vn/vn/the-gioi/quan-su/he-thong-phong-thu-ten-lua-tinh-vi-cua-israel-702784.html',
                  'https://vietnamnet.vn/vn/giao-duc/nguoi-thay/thu-tuong-can-doi-moi-mo-hinh-truong-chuyen-lop-chon-cho-phu-hop-hieu-qua-704079.html',
                  'https://vietnamnet.vn/vn/cong-nghe/cong-dong-mang/sau-ong-trump-twitter-xoa-nhieu-tai-khoan-ung-ho-cuu-tong-thong-my-704174.html',
                  'https://vietnamnet.vn/vn/suc-khoe/che-do-giam-can-bang-ca-phe-co-thuc-su-hieu-qua-701018.html']

    def parse(self, response):
        for next_page in  response.xpath('//a[@class="articletype_1"]/@href').getall():
            yield response.follow(next_page, self.parse, )

        body =  [x for x in response.css("p::text").getall() if x is not None and x != 'None' and len(x.split()) > 10 ]
        with open('/content/drive/MyDrive/nlp projects/Text correction/vietnamnet/rawtext.txt', 'a', encoding = 'utf-8') as f:
            f.writelines([para + ' ' for para in body])

def run_spider(spider):
    def f(q):
        try:
            runner = crawler.CrawlerRunner()
            deferred = runner.crawl(spider)
            deferred.addBoth(lambda _: reactor.stop())
            reactor.run()
            q.put(None)
        except Exception as e:
            q.put(e)

    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    result = q.get()
    p.join()

    if result is not None:
        raise result 

if __name__ == 'main':
    run_spider(VietnamnetSpider)
    lines = _read_file('/content/drive/MyDrive/nlp projects/Text correction/rawtext3.txt', 15, 40)
    with open('/content/drive/MyDrive/nlp projects/Text correction/rawtext3_lines_1540.txt', 'w', encoding = 'utf-8') as f:
    for line in lines:
        f.write(line   + '\n')