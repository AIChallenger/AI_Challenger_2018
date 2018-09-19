#!/usr/bin/perl
if(@ARGV > 0){
    print "usage: $0 < in > out\n";
    exit(1);
}

while(<STDIN>){
    chomp();
    if(m/<seg id=\"\d+\">\s*(.*)\s*<\/seg>/){
        print "$1\n";
    }
}
## End of main
