#!/usr/bin/perl -w
#  Description:
#    Segment the chinese string into a character sequence. Each word is seperated
#    by a space. Note that the word string should be segmented first by a Chinese
#    segmenter.


use Getopt::Long;

binmode STDIN,  ":utf8";
binmode STDOUT, ":utf8";
$|=1;

my $official = 0;
my $type = "plain";
GetOptions(
"t|type=s"  => \$type,
"h|help"    => \$help
) or die &usage;

if($help)
{
    &usage;
    exit;
}

while(<STDIN>)
{
    chomp();
    my $line = $_;
    my $text = $_;
    my $id   = -1;
    if($type eq "xml")
    {
        if($line =~ /<seg id="(.*)">\s*(.*)\s*<\/seg>/)
        {
            $id   = $1;
            $text = $2;
        }else{
            print $line."\n";
            next;
        }
    }

    $text = chn_char_segmentation($text);

    if($type eq "xml")
    {
        $line = "<seg id=\"$id\"> $text <\/seg>";
    }else{
        $line = $text;
    }

    print "$line\n";
}

## End of main

sub chn_char_segmentation
{
    my $text     = $_[0];
    my @words    = split /\s+/,$text;
    my $seg = "";
    foreach my $word (@words)
    {
            my @chars = ();
            my $tmp_str = $word;
            while($tmp_str){
                if($tmp_str =~ /^([^\x00-\x7f])/){ # any non-ascii word
                    push(@chars, $1);
                    $tmp_str =~ s/^.//;
                }elsif($tmp_str =~ /^([a-zA-Z]+)/){ # a english word
                    push(@chars, $1);
                    $tmp_str =~ s/^$1//;
                }elsif($tmp_str =~ /^(-?[\d,]+\.?\d+)/){ # a number
                    push(@chars, $1);
                    $tmp_str =~ s/^$1//;
                }elsif($tmp_str =~ /^(.)/){ # any char
                    push(@chars,$1);
                    $tmp_str =~ s/^.//;
                }
            }
            
            $seg .= " ".join(" ",@chars);
    }

    $seg =~ s/^\s*//;
    $seg =~ s/\s*$//;
    return $seg;
}



##################################################################################
sub usage
{
print <<__USAGE__;
usage: $0 -type plain < input > output
   -t,type     type of the input ( xml or plain )
   -h,help     print this help information
   
__USAGE__

}
